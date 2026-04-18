#include <bits/stdc++.h>
using namespace std;
mt19937 rng(random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
float dt=0.01;
int n=1000;
int d=8;
int r=64;
int deg=8;
//placeholders for now
float dh_chl_contrib=0.4f;
float dh_par_contrib=0.6f;
float fast_adaptation_rate=0.1f;
float slow_adaptation_learning_rate=0.01f;
float h_decay=0.3f;
float e_decay=0.1f;
float fixed_income=1.0f;
float cost_of_thought=0.5f;
float cost_of_complexity=0.5f;
//
vector<vector<int>> adj(n, vector<int>{}); //initialize with smallworld graph
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
            a[i][j]-=lr*err[i]*x[j];
            da[i][j]=-lr*err[i]*x[j]/dt;
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
        for (int j=0;j<d;j++){
            if (radj[i].size()>0) received_signal_par[j]/=radj[i].size();
            nerr[i][j]=received_signal_par[j]-h[i][j];
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
        //move h
        for (int j=0;j<d;j++){
            float fast_adapt=fast_adaptation_rate*(dh_par_contrib*nerr[i][j]-dh_chl_contrib*received_signal_chl[j]);
            float energy_noise=gaussian_noise(0.0f,1.0f/(max(0.0f,e[i])+0.1f));
            dh[i][j]=dt*(fast_adapt-h_decay*h[i][j]+energy_noise);
            nh[i][j]+=dh[i][j];
        }
        //move a
        delta_rule(na[i], emitted_signal_par, nerr[i], dt*slow_adaptation_learning_rate/(e[i]+0.1), da[i]); //need to weight this by energy etc.
        //update energy
        e[i]+=dt*(fixed_income/(1.0+surprise)-cost_of_thought*mag(dh[i])-cost_of_complexity*mag(da[i])-e_decay*e[i]);
        e[i]=max(0.0f, e[i]);
        u[i]=min(u[i],500.0f);
    }
    h=nh; err=nerr; a=na; u=nu;
}