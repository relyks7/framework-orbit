#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <thread>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <Accelerate/Accelerate.h>
using namespace std;
#define all(v) v.begin(), v.end()
using ll=long long;
int INF=0x3f3f3f3f;
float SENTINEL=-MAXFLOAT;
thread_local mt19937 rng(hash<thread::id>{}(this_thread::get_id())^random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
thread_local uniform_real_distribution<float> disf(0.0f, 1.0f);
float dt=0.01f;
void matvec(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int idx1, int idx2){
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*m), 1, 0.0f, c.data(), 1);
}
void matvec_transpose(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int idx1, int idx2){
    cblas_sgemv(CblasRowMajor, CblasTrans, n, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*n), 1, 0.0f, c.data(), 1);
}
void matmat(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int p, int idx1, int idx2){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, p, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*m*p), p, 0.0f, c.data(), p);
}
void matmat_transpose(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int p, int idx1, int idx2){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, p, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*p*m), m, 0.0f, c.data(), p);
}
float mag(const vector<float>&a, int start_idx, int size){
    return cblas_sdot(size, a.data()+start_idx, 1, a.data()+start_idx, 1);
}
void delta_rule(vector<float> &a, const vector<float>&x, const vector<float>&err, const vector<float>& y, float lr, float s, int n, int m, int idx1, int idx2){
    //a: nxm, x: mx1, err: nx1
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            a[i*m+j]+=-lr*dt*err[i]*(1-(y[idx2*n+i]*y[idx2*n+i])/(s*s))*x[idx1*m+j];
        }
    }
}
vector<float> randvec(int l, float mg){
    vector<float> ret(l, 0.0f); for (int i=0;i<l;i++) ret[i]=gaussian_noise(0, mg);
    return ret;
}
struct edge{
    vector<float> w;
    vector<float> u; 
    vector<float> v;
};
class lupus{
    public:
        int n, d;
        vector<unordered_map<int, edge>> adj;
        vector<float> h; //nxdx1
        vector<float> dh; //nxdx1
        vector<float> force; //nxdx1
        vector<float> received_signal; //dx1
        vector<float> chl_err; //dx1
        vector<float> scaled_dh; //dx1
        vector<float> par_change; //dx1
        vector<float> precision; //dx1
        vector<float> chl_err_p; //dx1
        vector<bool> fixed; //nx1
        float slow_learn, fast_learn, error_learn, eps, tanh_mag;
        void reset(){
            adj.assign(n, unordered_map<int, edge>{});
            adj[1][3]=edge{randvec(d*d, 1.0f/sqrtf(d)), vector<float>(d, 0.0f), vector<float>(d, 0.0f)};
            adj[3][0]=edge{randvec(d*d, 1.0f/sqrtf(d)), vector<float>(d, 0.0f), vector<float>(d, 0.0f)};
            adj[1][4]=adj[1][3];
            adj[4][2]=adj[3][0];
            h.assign(n*d, 0.0f); h=randvec(n*d, 1.0f/sqrtf(d));
            dh.assign(n*d, 0.0f);
            force.assign(n*d, 0.0f);
            received_signal.assign(d, 0.0f);
            chl_err.assign(d, 0.0f);
            scaled_dh.assign(d, 0.0f);
            par_change.assign(d, 0.0f);
            precision.assign(d, 0.0f);
            chl_err_p.assign(d, 0.0f);
            fixed.assign(n, false); fixed[0]=true; fixed[2]=true;
        }
        lupus(float un, float ud, float sl, float fl, float el, float e, float tm){
            n=un; d=ud;
            slow_learn=sl; fast_learn=fl; error_learn=el; eps=e; tanh_mag=tm;
            reset();
        }
        void forward(){
            fill(all(force), 0.0f);
            for (int par=0;par<n;par++){
                for (auto& [i, e]:adj[par]){
                    auto& [w, u, v]=e;
                    for (int j=0;j<d;j++) precision[j]=1.0f/(max(0.0f, u[j]-v[j]*v[j])+eps);
                    // for (int j=0;j<d;j++) precision[j]=1.0f;
                    matvec(w, h, received_signal, d, d, 0, par);
                    for (int j=0;j<d;j++) received_signal[j]=tanh_mag*tanhf(received_signal[j]/tanh_mag);
                    for (int j=0;j<d;j++) {
                        chl_err[j]=received_signal[j]-h[i*d+j];
                        chl_err_p[j]=chl_err[j]*precision[j];
                        force[i*d+j]+=chl_err_p[j];
                        u[j]+=dt*(chl_err[j]*chl_err[j]-u[j]);
                        v[j]+=dt*(chl_err[j]-v[j]);
                        scaled_dh[j]=dh[i*d+j]*(1-(received_signal[j]*received_signal[j])/(tanh_mag*tanh_mag));
                    }
                    matvec_transpose(w, scaled_dh, par_change, d, d, 0, 0);
                    for (int j=0;j<d;j++) force[par*d+j]-=par_change[j];
                    delta_rule(w, h, chl_err_p, received_signal, slow_learn, tanh_mag, d, d, par, 0);
                    if (par==1 && i==3) adj[1][4].w=w;
                    if (par==3 && i==0) adj[4][2].w=w;
                    if (par==1 && i==4) adj[1][3].w=w;
                    if (par==4 && i==2) adj[3][0].w=w;
                }
            }
            for (int i=0;i<n;i++){
                for (int j=0;j<d;j++) dh[i*d+j]+=dt*error_learn*(force[i*d+j]-dh[i*d+j]);
                if (!fixed[i]) for (int j=0;j<d;j++) h[i*d+j]+=dt*fast_learn*dh[i*d+j];
            }
        }
        vector<float> step(vector<float> ipt, vector<float> prior){
            vector<float> ret(d,0.0f);
            for (int i=0;i<d;i++) h[0*d+i]=ipt[i];
            for (int i=0;i<d;i++) h[2*d+i]=prior[i];
            forward();
            for (int i=0;i<d;i++) ret[i]+=dt*fast_learn*dh[0*d+i];
            return ret;
        }
};
vector<pair<float,float>> trial={{1.2f,0.4f},{-0.8f,1.1f},{0.3f,-1.4f},{-1.3f,-0.2f},{0.9f,1.0f},{-0.4f,-0.9f},{1.5f,-0.3f},{-1.0f,0.6f},{0.2f,1.6f},{-0.7f,-1.2f},{1.0f,-0.8f},{-1.5f,0.1f},{0.6f,0.7f},{-0.2f,-1.6f},{1.4f,0.5f},{-0.9f,-0.7f},{0.5f,-1.1f},{-1.2f,0.9f},{0.8f,-0.1f},{-0.3f,1.2f},{1.1f,-0.5f},{-0.6f,1.4f},{-1.4f,-0.5f},{0.4f,1.0f},{0.7f,-1.5f},{-1.1f,0.2f},{1.3f,0.8f},{-0.5f,-1.3f},{1.6f,0.0f},{0.0f,-1.0f}};
int total=1000; int succeeded=0;
vector<int> endsat(trial.size()+1,0);
vector<int> eachend(total, 0);
vector<lupus> sexti{};
float l1=1.0f, l2=1.0f;
float pival=3.141592653589793;
int main(){
    for (int _=0;_<total;_++){
        lupus sextus(5, 4, 0.01f, 15.0f, 4.0f, 1.0f, 8.0f);
        sexti.push_back(sextus);
        // float q1=-0.6f, q2=1.2f;
        // float cx=l1*cosf(q1)+l2*cosf(q1+q2);
        // float cy=l1*sinf(q1)+l2*sinf(q1+q2);
        float cx=0.0f, cy=0.0f;
        bool done=true;
        for (int i=0;i<trial.size();i++){
            auto [goalx, goaly]=trial[i];
            int timer=0;
            bool converged=false;
            for (int j=0;j<100000;j++){
                //float theta=pival/2.0f*tanhf(cx);
                float theta=0.25f*pival;
                vector<float> sense(sextus.d,0.0f);
                sense[0]=cx; sense[1]=cy; 
                //sense[2]=cosf(theta); sense[3]=sinf(theta);
                vector<float> want(sextus.d,0.0f);
                want[0]=goalx; want[1]=goaly;
                vector<float> mv=sextus.step(sense, want);
                // q1+=mv[0]; q2+=mv[1];
                // q1=remainderf(q1, 2.0f*pival);
                // q2=remainderf(q2, 2.0f*pival);
                // cx=l1*cosf(q1)+l2*cosf(q1+q2);
                // cy=l1*sinf(q1)+l2*sinf(q1+q2);
                cx+=mv[0]*cosf(theta)-mv[1]*sinf(theta);
                cy+=mv[0]*sinf(theta)+mv[1]*cosf(theta);
                // cx+=mv[0];
                if (abs(cx-goalx)<0.01f && abs(cy-goaly)<0.01f) timer++;
                else timer=0;
                if (timer>200) {
                    converged=true; break;
                }
            }
            if (!converged){
                done=false;
                endsat[i]++;
                eachend[_]=i;
                break;
            }
        }
        if (done) {
            endsat[trial.size()]++;
            eachend[_]=trial.size();
            succeeded++;
        }
    }
    for (int i=0;i<=trial.size();i++) cout<<i<<": "<<endsat[i]<<'\n';
    cout<<succeeded<<'/'<<total<<'\n'<<setprecision(4)<<100.0f*succeeded/total<<"%\n";
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/