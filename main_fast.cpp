#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <thread>
#include <iomanip>
#include <unordered_map>
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
vector<float> topk(vector<float> a, int k){
    vector<int> inds(a.size(), 0); iota(all(inds), 0);
    nth_element(inds.begin(), inds.begin()+k, inds.end(), [&](const int& xx, const int& yy){return fabs(a[xx])>fabs(a[yy]);});
    for (int i=k;i<a.size();i++) a[inds[i]]=0.0f;
    return a;
}
void delta_rule(vector<float> &a, const vector<float>&x, const vector<float>&err, float lr, int n, int m, int idx){
    //a: nxm, x: mx1, err: nx1
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            a[i*m+j]+=-lr*dt*err[i]*x[idx*m+j];
        }
    }
}
class lupus{
    public:
        int n, d, k;
        vector<unordered_map<int, vector<float>>> radj;
        vector<float> h; //nxdx1
        vector<float> dh; //nxdx1
        vector<float> err; //dx1
        vector<float> m; //nxdxd
        vector<float> u; //nxscalar
        vector<float> received_signal; //dx1
        vector<float> chl_err; //dx1
        vector<float> par_change; //dx1
        vector<float> m_t; //dxd
        float slow_learn, fast_learn;
        float omega, tau;
        void forward(){
            fill(all(dh), 0.0f);
            for (int i=0;i<n;i++){
                fill(all(err), 0.0f);
                fill(all(m_t), 0.0f);
                float err_mag=0.0f;
                for (auto& [par, w]:radj[i]){
                    matvec(w, h, received_signal, d, d, 0, par);
                    for (int j=0;j<d;j++) {
                        chl_err[j]=received_signal[j]-h[i*d+j];
                        err[j]+=chl_err[j];
                    }
                    for (int j=0;j<d;j++) {
                        for (int l=0;l<d;l++){
                            m_t[j*d+l]+=chl_err[j]*chl_err[l]/radj[i].size();
                        }
                    }
                    err_mag+=mag(chl_err, 0, d)/(d*radj[i].size());
                    matvec_transpose(w, chl_err, par_change, d, d, 0, 0);
                    for (int j=0;j<d;j++) dh[par*d+j]-=par_change[j];
                    delta_rule(w, h, chl_err, slow_learn, d, d, par);
                }
                u[i]+=dt*(err_mag-u[i]);
                for (int j=0;j<d;j++) {
                    for (int l=0;l<d;l++){
                        m[i*d*d+j*d+l]+=dt*(m_t[j*d+l]-m[i*d*d+j*d+l]);
                    }
                }
                for (int j=0;j<d;j++) dh[i*d+j]+=err[j];
            }
            for (int i=0;i<n;i++){
                vector<float> idh(d,0);
                for (int j=0;j<d;j++){
                    idh[j]=dh[i*d+j];
                }
                idh=topk(idh, k);
                for (int j=0;j<d;j++) h[i*d+j]+=dt*fast_learn*idh[j];
            }
        }
};