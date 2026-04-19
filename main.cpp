#include <bits/stdc++.h>
using namespace std;
mt19937 rng(random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
uniform_real_distribution<float> disf(0.0f, 1.0f);
float dt=0.01;
int n=50;
int d=8;
int r=64;
int deg=8;
float dh_chl_contrib=1.0f;
float dh_par_contrib=0.6f;
float fast_adaptation_rate=10.0f;
float slow_adaptation_learning_rate=0.005f;
float h_decay=0.05f;
float e_decay=0.01f;
float signal_income=0.2f;
float cost_of_thought=0.1f;
float cost_of_complexity=1.0f;
float curiosity=0.5f;
float stability=0.9f;
float surprise_tax=0.5; //"surprise tax" would be terrible in any other context
float a_decay=0.05f;
vector<vector<int>> adj(n, vector<int>{}); //initialize with smallworld graph
vector<bool> input(n,false);
vector<vector<int>> radj(n, vector<int>{});
vector<vector<float>> h(n, vector<float>(d,0)); //internal state
vector<vector<float>> dh(n, vector<float>(d,0));
vector<float> e(n, 0); //energy
vector<vector<float>> err(n, vector<float>(d,0)); //error
vector<float> u(n, 0); //uncertainty
vector<vector<vector<float>>> a(n,vector<vector<float>>(d,vector<float>(r,0))); //receptors
vector<vector<vector<float>>> da(n,vector<vector<float>>(d,vector<float>(r,0)));
vector<vector<vector<float>>> b(n,vector<vector<float>>(d,vector<float>(r,0))); //emitters
vector<float> matvec(const vector<vector<float>>& a, const vector<float>& b){
    int n=a.size();
    int m=a[0].size();
    vector<float> c(n,0);
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            c[i]+=a[i][j]*b[j];
        }
    }
    return c;
}
//note: no transposed matvec, because from previous experience gpu is better with the operations being separate
vector<vector<float>> transpose(const vector<vector<float>>& a){
    int n=a.size();
    int m=a[0].size();
    vector<vector<float>> a_t(m,vector<float>(n,0));
    for (int i=0;i<m;i++){
        for (int j=0;j<n;j++){
            a_t[i][j]=a[j][i];
        }
    }
    return a_t;
}
vector<vector<float>> outer_prod(const vector<float>& a, const vector<float>& b){
    int n=a.size();
    int m=b.size();
    vector<vector<float>> c(n, vector<float>(m, 0));
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            c[i][j]=a[i]*b[j];
        }
    }
    return c;
}
void delta_rule(vector<vector<float>> &a, const vector<float>&x, const vector<float>&err, float lr, vector<vector<float>>& da){
    for (int i=0;i<a.size();i++){
        for (int j=0;j<a[0].size();j++){
            da[i][j]=-lr*dt*err[i]*x[j];
            a[i][j]+=-dt*a_decay*a[i][j]-da[i][j];
        }
    }
}
float mag(const vector<float>&a){
    float ret=0;
    for (auto i:a) ret+=i*i;
    return ret;
}
float mag(const vector<vector<float>>&a){
    float ret=0;
    for (auto i:a) for (auto x:i) ret+=x*x;
    return ret;
}
void generate_smallworld(float p){
    uniform_int_distribution<int> disi(0,n-2);
    for (int i=0;i<n;i++){
        for (int j=-(deg/2);j<=(deg/2);j++){
            if (j==0) continue;
            if (disf(rng)>p){
                adj[i].push_back((i+j+n)%n); radj[(i+j+n)%n].push_back(i);
            } else{
                int dart=disi(rng);
                if (dart>=i) dart++;
                adj[i].push_back(dart); radj[dart].push_back(i);
            }
        }
    }
}
float curx=0.25f, cury=0.25f, foodx=0.5f, foody=0.5f, bound=1.0f, fr=0.1, speed=2.0f;
float hunger=1.0f;
void update(){
    vector<float> nu=u;
    vector<vector<float>> nh=h;
    vector<vector<float>> nerr=err;
    vector<vector<vector<float>>> na=a;
    for (int i=0;i<n;i++){
        //step 1: aggregate belief from parents
        vector<float> emitted_signal_par(r,0);
        for (auto par:radj[i]){
            vector<float> par_signal=matvec(transpose(b[par]), h[par]);
            for (int j=0;j<r;j++){
                emitted_signal_par[j]+=1.0f/(1.0f+u[par])*par_signal[j];
            }
        }
        vector<float> received_signal_par=matvec(a[i], emitted_signal_par);
        float surprise=0.0f; //surprise=squared sum of error
        float sig_mag=0.0f; //magnitude of received signal
        for (int j=0;j<d;j++){
            if (radj[i].size()>0) received_signal_par[j]/=radj[i].size();
            nerr[i][j]=received_signal_par[j]-h[i][j];
            sig_mag+=received_signal_par[j]*received_signal_par[j];
            //XXX: temporary addon to add hunger prior
            if (i==0){
                nerr[i][j]=((1.0f-hunger)+sqrtf(pow(curx-foodx, 2)+pow(cury-foody, 2)))*2.0f;
            }
            surprise+=nerr[i][j]*nerr[i][j];
        }
        nu[i]+=dt*((surprise/d)-u[i]); //uncertainty is a moving average of surprise
        //step 2: aggregate error from children
        vector<float> emitted_signal_chl(r,0);
        for (auto chl:adj[i]){
            vector<float> chl_signal=matvec(transpose(a[chl]), err[chl]);
            for (int j=0;j<r;j++){
                emitted_signal_chl[j]+=1.0f/(1.0f+u[chl])*chl_signal[j];
            }
        }
        vector<float> received_signal_chl=matvec(b[i], emitted_signal_chl);
        for (int j=0;j<d;j++){
            if (adj[i].size()>0) received_signal_chl[j]/=adj[i].size();
            sig_mag+=received_signal_chl[j]*received_signal_chl[j];
        }
        //move h
        for (int j=0;j<d;j++){
            if (input[i]) continue;
            float fast_adapt=fast_adaptation_rate*(dh_par_contrib*nerr[i][j]-dh_chl_contrib*received_signal_chl[j]);
            float energy_noise=gaussian_noise(0.0f,1.0f/(max(0.0f,e[i])+0.1f));
            dh[i][j]=dt*(fast_adapt-h_decay*h[i][j]+energy_noise);
            nh[i][j]+=dh[i][j];
        }
        //move a
        delta_rule(na[i], emitted_signal_par, nerr[i], slow_adaptation_learning_rate/(e[i]+0.1), da[i]); //need to weight this by energy etc.
        //update energy
        e[i]+=dt*(1.0/(1.0f+u[i])*(curiosity*signal_income*sig_mag-stability*surprise_tax*surprise)-cost_of_thought*mag(dh[i])-cost_of_complexity*mag(da[i])-e_decay*e[i]);
        e[i]=max(0.0f, e[i]);
        nu[i]=min(nu[i],10.0f);
    }
    h=nh; err=nerr; a=na; u=nu;
}
int main(){
    generate_smallworld(0.1f);
    for (int i=0;i<n;i++){
        for (int j=0;j<d;j++){
            for (int k=0;k<r;k++){
                a[i][j][k]=gaussian_noise(0.0f, 0.2f);
                b[i][j][k]=gaussian_noise(0.0f, 0.2f);
            }
        }
    }
    for (int i=0;i<50000;i++){
        for(int j=1;j<=4;j++) input[j]=true;
        fill(dh[1].begin(), dh[1].end(), curx-h[1][0]);
        fill(h[1].begin(), h[1].end(), curx);
        fill(dh[2].begin(), dh[2].end(), cury-h[2][0]);
        fill(h[2].begin(), h[2].end(), cury);
        fill(dh[3].begin(), dh[3].end(), foodx-h[3][0]);
        fill(h[3].begin(), h[3].end(), foodx);
        fill(dh[4].begin(), dh[4].end(), foody-h[4][0]);
        fill(h[4].begin(), h[4].end(), foody);
        update();
        curx+=dt*speed*tanhf(h[5][0]);
        cury+=dt*speed*tanhf(h[5][1]);
        curx=max(0.0f,min(bound,curx));
        cury=max(0.0f,min(bound,cury));
        if (sqrtf(pow(curx-foodx, 2)+pow(cury-foody, 2))<fr){
            hunger+=0.3;
            foodx=disf(rng)*bound;
            foody=disf(rng)*bound;
        }
        hunger-=0.01*dt;
        hunger=max(0.0f,min(1.0f,hunger));
        if (i%50==0){
            cout<<setprecision(3)<<"tick: "<<setw(6)<<i<<"|curx:"<<setw(6)<<curx<<"|cury: "<<setw(6)<<cury<<"|foodx: "<<setw(6)<<foodx<<"|foody: "<<setw(6)<<foody<<"|hunger: "<<setw(6)<<hunger<<endl;
        }
    }
    return 0;
}