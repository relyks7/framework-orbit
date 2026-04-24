#include <bits/stdc++.h>
using namespace std;
mt19937 rng(random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
uniform_real_distribution<float> disf(0.0f, 1.0f);
float dt=0.01f;
int n=100;
int d=16;
int r=64;
int deg=16;
float starting_energy=0.5f;
float fr=0.05f; //food radius
float speed=2.0f;
float bound=1.0f;
float within_dist=0.2f; //tolerance for differentiating direcion
float phi=1.618033988749895;
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
float mag(const vector<float>&a){
    float ret=0;
    for (auto i:a) ret+=i*i;
    return ret/a.size();
}
float mag(const vector<vector<float>>&a){
    float ret=0;
    for (auto i:a) for (auto x:i) ret+=x*x;
    return ret/(a.size()*a[0].size());
}
void delta_rule(vector<vector<float>> &a, const vector<float>&x, const vector<float>&err, float lr, vector<vector<float>>& da, float a_dec){
    for (int i=0;i<a.size();i++){
        for (int j=0;j<a[0].size();j++){
            da[i][j]=-lr*dt*err[i]*x[j];
            a[i][j]+=-dt*a_dec*a[i][j]-da[i][j];
        }
    }
}
void generate_smallworld(float p, vector<vector<int>>& iadj, vector<vector<int>>& iradj){
    uniform_int_distribution<int> disi(0,n-2);
    for (int i=0;i<n;i++){
        for (int j=-(deg/2);j<=(deg/2);j++){
            if (j==0) continue;
            if (disf(rng)>p){
                iadj[i].push_back((i+j+n)%n); iradj[(i+j+n)%n].push_back(i);
            } else{
                int dart=disi(rng);
                if (dart>=i) dart++;
                iadj[i].push_back(dart); iradj[dart].push_back(i);
            }
        }
    }
}
struct genes{
    float dh_chl_contrib;
    float dh_par_contrib;
    float fast_adaptation_rate;
    float slow_adaptation_learning_rate_a;
    float slow_adaptation_learning_rate_b; //a bit larger due to it's reception of error requiring more volatile shift
    float h_decay;
    float e_decay;
    float signal_income;
    float cost_of_thought;
    float cost_of_complexity;
    float curiosity;
    float stability;
    float surprise_tax; //"surprise tax" would be a terrible phrase in any other context
    float a_decay;
    float b_decay;
};
// float dh_chl_contrib=1.0f;
// float dh_par_contrib=0.6f;
// float fast_adaptation_rate=10.0f;
// float slow_adaptation_learning_rate_a=0.005f;
// float slow_adaptation_learning_rate_b=0.0075f; //a bit larger due to it's reception of error requiring more volatile shift
// float h_decay=0.05f;
// float e_decay=0.01f;
// float signal_income=0.2f;
// float cost_of_thought=0.1f;
// float cost_of_complexity=1.0f;
// float curiosity=0.5f;
// float stability=0.9f;
// float surprise_tax=0.5f; //"surprise tax" would be a terrible phrase in any other context
// float a_decay=0.05f;
class lupus{
    //0 is prior, 1 is eyes, 2 is motor
    public:
        vector<vector<int>> adj; //initialize with smallworld graph
        vector<bool> input;
        vector<vector<int>> radj;
        vector<vector<float>> h; //internal state
        vector<vector<float>> dh;
        vector<float> e; //energy, initially high
        vector<vector<float>> err; //error
        vector<float> u; //uncertainty
        vector<vector<vector<float>>> a; //receptors
        vector<vector<vector<float>>> da;
        vector<vector<vector<float>>> b; //emitters
        vector<vector<vector<float>>> db;
        genes dna;
        float curx;
        float cury;
        float foodx;
        float foody;
        float hunger;
        int curdir;
        bool alive;
        lupus(const vector<vector<int>>& uadj, const vector<vector<int>>& uradj, const genes& rdna, const vector<bool>& uinput){
            adj=uadj; radj=uradj; dna=rdna; input=uinput;
            h.assign(n, vector<float>(d,0));
            dh.assign(n, vector<float>(d,0));
            e.assign(n, starting_energy);
            err.assign(n, vector<float>(d,0));
            a.assign(n,vector<vector<float>>(d,vector<float>(r,0)));
            da.assign(n,vector<vector<float>>(d,vector<float>(r,0)));
            b.assign(n,vector<vector<float>>(d,vector<float>(r,0)));
            db.assign(n,vector<vector<float>>(d,vector<float>(r,0)));
        }
        void reset(){
            alive=true;
            for (int i=0;i<n;i++){
                for (int j=0;j<d;j++){
                    for (int k=0;k<r;k++){
                        a[i][j][k]=gaussian_noise(0.0f, 0.2f);
                        b[i][j][k]=gaussian_noise(0.0f, 0.2f);
                    }
                }
            }
            curx=0.25f;
            cury=0.25f;
            foodx=0.5f;
            foody=0.5f;
            hunger=1.0f;
        }
        void update(){
            vector<float> nu=u;
            vector<vector<float>> nh=h;
            vector<vector<float>> nerr=err;
            vector<vector<vector<float>>> na=a;
            vector<vector<vector<float>>> nb=b;
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
                for (int j=0;j<d;j++){
                    if (radj[i].size()>0) received_signal_par[j]/=radj[i].size();
                    nerr[i][j]=received_signal_par[j]-h[i][j];
                    //XXX: temporary addon to add hunger prior
                    if (i==0 && j==0){
                        nerr[i][j]=((1.0f-hunger)*sqrtf(pow(curx-foodx, 2)+pow(cury-foody, 2)))*phi;
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
                }
                float sig_mag=mag(received_signal_par)+mag(received_signal_chl); //magnitude of received signal
                //move h
                for (int j=0;j<d;j++){
                    if (input[i]) continue;
                    float fast_adapt=dna.fast_adaptation_rate*(dna.dh_par_contrib*nerr[i][j]-dna.dh_chl_contrib*received_signal_chl[j]);
                    float energy_noise=gaussian_noise(0.0f,1.0f/(max(0.0f,e[i])+0.1f));
                    dh[i][j]=dt*(fast_adapt-dna.h_decay*h[i][j]+energy_noise);
                    nh[i][j]+=dh[i][j];
                }
                //move a
                delta_rule(na[i], emitted_signal_par, nerr[i], dna.slow_adaptation_learning_rate_a/(e[i]+0.5), da[i], dna.a_decay);
                //move b (hypothesis here - align the errors, invert A=sort of "unify/harmonize" the whole network)
                delta_rule(nb[i], emitted_signal_chl, nerr[i], dna.slow_adaptation_learning_rate_b/(e[i]+0.5), db[i], dna.b_decay);
                //update energy
                e[i]+=dt*(1.0/(1.0f+u[i])*(dna.curiosity*dna.signal_income*sig_mag-dna.stability*dna.surprise_tax*surprise)-dna.cost_of_thought*mag(dh[i])-dna.cost_of_complexity*mag(da[i])-dna.e_decay*e[i]);
                e[i]=max(0.0f, e[i]);
                nu[i]=min(nu[i],10.0f);
            }
            h=nh; err=nerr; a=na; b=nb; u=nu;
        }
        void step(){
            float disx=foodx-curx;
            float disy=foody-cury;
            int disxi, disyi;
            if (disx<-within_dist) disxi=0;
            else if (disx>within_dist) disxi=2;
            else disxi=1;
            if (disy<-within_dist) disyi=0;
            else if (disy>within_dist) disyi=2;
            else disyi=1;
            curdir=disxi*3+disyi+1;
            for (int i=0;i<16;i++){
                if (i==curdir){
                    dh[1][i]=1.0f-h[1][i];
                    h[1][i]=1.0f;
                } else{
                    dh[1][i]=0.0f-h[1][i];
                    h[1][i]=0.0f;
                }
            }
            update();
            curx+=dt*speed*tanhf(h[2][0]);
            cury+=dt*speed*tanhf(h[2][1]);
            curx=max(0.0f,min(bound,curx));
            cury=max(0.0f,min(bound,cury));
            if (sqrtf(pow(curx-foodx, 2)+pow(cury-foody, 2))<fr){
                hunger+=0.3;
                foodx=disf(rng)*bound;
                foody=disf(rng)*bound;
            }
            hunger-=0.01*dt;
            hunger=max(0.0f,min(1.0f,hunger));
            if (hunger==0.0f) alive=false;
        }
};
vector<vector<int>> un_adj(n,vector<int>{});
vector<vector<int>> un_radj(n,vector<int>{});
vector<float> un_input(n, false);
int main(){
    generate_smallworld(0.1f, un_adj, un_radj);
    un_input[1]=true;
    return 0;
}