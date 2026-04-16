#include <bits/stdc++.h>
using namespace std;
int n=1000;
int d=8;
int r=64;
int dt=0.01;
int deg=8;
vector<vector<int>> adj(n, vector<int>(deg,-1)); //initialize with smallworld graph
vector<vector<int>> radj(n, vector<int>(deg,-1));
vector<vector<float>> h(n, vector<float>(d,0)); //internal state
vector<float> e(n, 0); //energy
vector<vector<float>> err(n, vector<float>(d,0)); //error
vector<float> u(n, 0); //uncertainty
vector<vector<vector<float>>> a(n,vector<vector<float>>(d,vector<float>(r,0))); //receptors
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
void update(){
    for (int i=0;i<n;i++){
        //step 1: aggregate belief from parents. natch current state to that belief via a.
        vector<float> emitted_signal(r,0);
        for (auto par:radj[i]){
            vector<float> par_signal=matvec(transpose(b[par]), h[par]);
            for (int j=0;j<r;j++){
                emitted_signal[j]+=1.0f/(1.0f+u[par])*tanhf(par_signal[j]);
            }
        }
        vector<float> received_signal=matvec(a[i], emitted_signal);
        for (int j=0;j<r;j++){
            received_signal[j]/=deg;
        }
    }
}