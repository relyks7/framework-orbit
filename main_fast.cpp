#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <thread>
#include <iomanip>
#include <unordered_set>
#include <Accelerate/Accelerate.h>
using namespace std;
#define all(v) v.begin(), v.end()
using ll=long long;
int INF=0x3f3f3f3f;
thread_local mt19937 rng(hash<thread::id>{}(this_thread::get_id())^random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
thread_local uniform_real_distribution<float> disf(0.0f, 1.0f);
float dt=0.01f;
float tanh_mag=10.0f;
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
void delta_rule(vector<float> &a, const vector<float>&x, const vector<float>&err, float lr, vector<float>& da, int n, int m, int idx){
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            da[idx*m*n+i*m+j]=-lr*dt*err[idx*n+i]*x[j];
            a[idx*m*n+i*m+j]+=da[idx*m*n+i*m+j];
        }
    }
}
class lupus{
    //0 is prior, 1 is eyes, 2 is motor
    public:
        vector<unordered_set<int>> adj;
        vector<bool> input;
        vector<unordered_set<int>> radj;
        vector<float> h; //internal state
        vector<float> dh;
        vector<float> err; //error
        vector<float> err_diff;
        vector<float> err_cov; //error covariance matrix
        vector<float> u; //free energy
        vector<float> a; //receptors
        vector<float> da;
        vector<float> b; //emitters
        vector<float> db;
        //scratch
        vector<float> par_signal;
        vector<float> emitted_signal_par;
        vector<float> received_signal_par;
        vector<float> nu, nh, nerr, nerr_diff;
        vector<float> emitted_signal_chl;
        vector<float> chl_signal;
        vector<float> received_signal_chl;
        vector<float> pred_h;
        int ticks=0;
        int cticks;
        int n=2;
        int d=4;
        int r=32;
        float fast_learn;
        float slow_learn;
        float tau;
        float omega;
        lupus(const vector<bool>& uinput, int icticks){
            adj.assign(n,unordered_set<int>{}); radj.assign(n,unordered_set<int>{}); input=uinput;
            h.assign(n*d, 0.0f);
            dh.assign(n*d, 0.0f);
            err.assign(n*d, 0.0f);
            err_cov.assign(n*d*d, 0.0f);
            err_diff.assign(n*d, 0.0f);
            u.assign(n, 0.0f);
            a.assign(n*d*r, 0.0f);
            da.assign(n*d*r, 0.0f);
            b.assign(n*d*r, 0.0f);
            db.assign(n*d*r, 0.0f);
            par_signal.assign(r, 0.0f);
            pred_h.assign(d,0);
            emitted_signal_par.assign(r, 0.0f);
            received_signal_par.assign(d, 0.0f);
            chl_signal.assign(r, 0.0f);
            emitted_signal_chl.assign(r, 0.0f);
            received_signal_chl.assign(d, 0.0f);
            cticks=icticks;
            fast_learn=1.0f; slow_learn=0.02f;
            tau=0.12f; omega=0.25f;
        }
        void cleanup(){ //apoptosis, run every k ticks
            vector<bool> alive(n,true);
            vector<int> new_index(n,-1);
            int cnt=0;
            for (int i=0;i<n;i++){
                if (!input[i] && u[i]>omega){
                    alive[i]=false; 
                    cout<<"NODE "<<i<<" DEAD IN CLEANUP"<<endl;
                    continue;
                }
                new_index[i]=cnt++;
            }
            nh.clear();
            vector<float> ndh{};
            nerr.clear();
            nerr_diff.clear();
            nu.clear();
            vector<float> na{};
            vector<float> nda{};
            vector<float> nb{};
            vector<float> ndb{};
            vector<float> nerr_cov{};
            vector<bool> ninput;
            for (int i=0;i<n;i++){
                if (alive[i]){
                    ninput.push_back(input[i]);
                    nu.push_back(u[i]);
                    for (int j=0;j<d;j++){
                        nh.push_back(h[i*d+j]);
                        ndh.push_back(dh[i*d+j]);
                        nerr.push_back(err[i*d+j]);
                        nerr_diff.push_back(err_diff[i*d+j]);
                    }
                    for (int j=0;j<d*r;j++){
                        na.push_back(a[i*d*r+j]);
                        nda.push_back(da[i*d*r+j]);
                        nb.push_back(b[i*d*r+j]);
                        ndb.push_back(db[i*d*r+j]);
                    }
                    for (int j=0;j<d*d;j++){
                        nerr_cov.push_back(err_cov[i*d*d+j]);
                    }
                }
            }
            swap(h, nh);
            swap(dh, ndh);
            swap(err, nerr);
            swap(err_diff, nerr_diff);
            swap(u, nu);
            swap(a, na);
            swap(da, nda);
            swap(b, nb);
            swap(db, ndb);
            swap(err_cov, nerr_cov);
            swap(input, ninput);
            vector<unordered_set<int>> nadj{};
            vector<unordered_set<int>> nradj{};
            for (int i=0;i<n;i++){
                if (!alive[i]) continue;
                nadj.push_back(unordered_set<int>{});
                nradj.push_back(unordered_set<int>{});
                for (auto chl:adj[i]){
                    if (alive[chl]) nadj[new_index[i]].insert(new_index[chl]);
                }
                for (auto par:radj[i]){
                    if (alive[par]) nradj[new_index[i]].insert(new_index[par]);
                }
            }
            swap(adj, nadj);
            swap(radj, nradj);
            n=cnt;
        }
        void mitosis(int i){ //no, kris did not mitose
            cout<<"mitosis at node "<<i<<endl;
            vector<float> cov(d*d);
            for (int j=0;j<d*d;j++) cov[j]=err_cov[i*d*d+j];
            vector<float> eigenvals(d,0.0f);
            int lwork=3*d-1;
            vector<float> work(lwork, 0.0f);
            int info=0;
            char jobz='V'; //eigenval and eigenvec
            char uplo='U'; //upper triangle (cov is symmetric)
            ssyev_(&jobz, &uplo, &d, cov.data(), &d, eigenvals.data(), work.data(), &lwork, &info);
            if (info!=0){
                cout<<"eigendecomposition error, info: "<<info<<endl;
                cout<<"error: "<<((info>0)?"did not converge":"illegal parameter")<<endl;
                return;
            }
            vector<float> eigenvec{};
            for (int j=0;j<d;j++){
                eigenvec.push_back(cov[d*(d-1)+j]);
            }
            vector<float> c1_h(d, 0.0f);
            vector<float> c2_h(d, 0.0f);
            vector<float> c1_a(d*r, 0.0f);
            vector<float> c2_a(d*r, 0.0f);
            vector<float> c1_b(d*r, 0.0f);
            vector<float> c2_b(d*r, 0.0f);
            float temp=0.0f;
            for (int j=0;j<d;j++){
                temp+=h[i*d+j]*eigenvec[j];
            }
            for (int j=0;j<d;j++){
                c1_h[j]=eigenvec[j]*temp;
                c2_h[j]=h[i*d+j]-c1_h[j];
            }
            vector<float> temp0(r,0.0f);
            matmat(eigenvec, a, temp0, 1, d, r, 0, i);
            matmat(eigenvec, temp0, c1_a, d, 1, r, 0, 0);
            matmat(eigenvec, b, temp0, 1, d, r, 0, i);
            matmat(eigenvec, temp0, c1_b, d, 1, r, 0, 0);
            for (int j=0;j<d*r;j++){
                c2_a[j]=a[i*d*r+j]-c1_a[j];
                c2_b[j]=b[i*d*r+j]-c1_b[j];
            }
            //final ordering: c1 takes the place of i, aggregator second last, c2 last
            //for aggregator we want A_agg@B^t=I, so B@A_agg^T=I
            vector<float> scratch(d*r,0);
            for (int j=0;j<r;j++) for (int k=0;k<d;k++) scratch[j*d+k]=b[i*d*r+k*r+j];
            vector<float> scratch_copy=scratch;
            vector<float> target(d*r,0); for (int j=0;j<d;j++) target[j*r+j]=1; //I^d
            vector<float> s(d);
            float eps=1e-5f;
            int rank; int ls_info; float opt_work; int lwork_q=-1;
            vector<int> iwork(128);
            sgelsd_(&d, &r, &d, scratch_copy.data(), &d, target.data(), &r, s.data(), &eps, &rank, &opt_work, &lwork_q, iwork.data(), &ls_info);
            int ls_lwork=(int)opt_work;
            vector<float> ls_work(ls_lwork);
            sgelsd_(&d, &r, &d, scratch_copy.data(), &d, target.data(), &r, s.data(), &eps, &rank, ls_work.data(), &ls_lwork, iwork.data(), &ls_info);
            if (ls_info!=0) cout<<"error in predict (sgelsd_), info: "<<ls_info<<'\n';
            vector<float> a_agg=target;
            for (int j=0;j<d*r;j++){
                a_agg[j]*=2.0f; //correct for mean
            }
            for (int j=0;j<d;j++){
                h.push_back(h[i*d+j]);
                h[i*d+j]=c1_h[j];
                dh.push_back(0.0f);
                dh[i*d+j]=0.0f;
                err.push_back(0.0f);
                err[i*d+j]=0.0f;
                err_diff.push_back(0.0f);
                err_diff[i*d+j]=0.0f;
                for (int k=0;k<d;k++){
                    err_cov.push_back(0.0f);
                    err_cov[i*d*d+j*d+k]=0.0f;
                }
                for (int k=0;k<r;k++){
                    a.push_back(a_agg[j*r+k]);
                    b.push_back(b[i*d*r+j*r+k]);
                    a[i*d*r+j*r+k]=c1_a[j*r+k];
                    b[i*d*r+j*r+k]=c1_b[j*r+k];
                    da.push_back(0.0f);
                    db.push_back(0.0f);
                    da[i*d*r+j*r+k]=0.0f;
                    db[i*d*r+j*r+k]=0.0f;
                }
            }
            for (int j=0;j<d;j++){
                h.push_back(c2_h[j]);
                dh.push_back(0.0f);
                err.push_back(0.0f);
                err_diff.push_back(0.0f);
                for (int k=0;k<d;k++){
                    err_cov.push_back(0.0f);
                }
                for (int k=0;k<r;k++){
                    a.push_back(c2_a[j*r+k]);
                    b.push_back(c2_b[j*r+k]);
                    da.push_back(0.0f);
                    db.push_back(0.0f);
                }
            }
            u.push_back(0.0f);
            u.push_back(0.0f);
            u[i]=0.0f;
            input.push_back(false);
            input.push_back(false);
            adj.push_back({});
            adj.push_back({});
            radj.push_back({});
            radj.push_back({});
            n+=2;
            for (auto par:radj[i]){
                adj[par].insert(n-1);
                radj[n-1].insert(par);
            }
            for (auto chl:adj[i]){
                adj[n-2].insert(chl);
                radj[chl].erase(i);
                radj[chl].insert(n-2);
            }
            adj[i]={n-2};
            adj[n-1]={n-2};
            radj[n-2]={i, n-1};
        }
        void reset(){
            n=2;
            adj.assign(n,unordered_set<int>{}); radj.assign(n,unordered_set<int>{});
            if (input.size()>n){
                input.erase(input.begin()+n, input.end());
            }
            pred_h.assign(d,0);
            h.assign(n*d, 0.0f);
            dh.assign(n*d, 0.0f);
            err.assign(n*d, 0.0f);
            err_cov.assign(n*d*d, 0.0f);
            err_diff.assign(n*d, 0.0f);
            u.assign(n, 0.0f);
            a.assign(n*d*r, 0.0f);
            da.assign(n*d*r, 0.0f);
            b.assign(n*d*r, 0.0f);
            db.assign(n*d*r, 0.0f);
            par_signal.assign(r, 0.0f);
            emitted_signal_par.assign(r, 0.0f);
            received_signal_par.assign(d, 0.0f);
            chl_signal.assign(r, 0.0f);
            emitted_signal_chl.assign(r, 0.0f);
            received_signal_chl.assign(d, 0.0f);
            ticks=0;
            adj.assign(n,unordered_set<int>{}); radj.assign(n,unordered_set<int>{});
            for (int i=0;i<n;i++){
                for (int j=0;j<d;j++){
                    for (int k=0;k<r;k++){
                        a[i*d*r+j*r+k]=gaussian_noise(0.0f, 1.0f/sqrtf(r));
                        b[i*d*r+j*r+k]=gaussian_noise(0.0f, 1.0f/sqrtf(r));
                    }
                }
            }
            adj[0].insert(1);
            adj[1].insert(0);
            radj[0].insert(1);
            radj[1].insert(0);
            // adj[0].insert(1);
            // adj[0].insert(2);
            // adj[1].insert(3);
            // adj[2].insert(3);
            // adj[3].insert(0);
            // radj[1].insert(0);
            // radj[2].insert(0);
            // radj[3].insert(1);
            // radj[3].insert(2);
            // radj[0].insert(3);
        }
        void forward(){
            nu=u;
            nh=h;
            nerr=err;
            nerr_diff=err_diff;
            for (int i=0;i<n;i++){
                //step 1: aggregate belief from parents
                fill(all(emitted_signal_par), 0.0f);
                for (auto par:radj[i]){
                    matvec_transpose(b, h, par_signal, d, r, par, par);
                    for (int j=0;j<r;j++){
                        emitted_signal_par[j]+=tanh_mag*tanhf(par_signal[j]/tanh_mag)/radj[i].size();
                    }
                }
                matvec(a, emitted_signal_par, received_signal_par ,d, r, i, 0);
                for (int j=0;j<d;j++){
                    received_signal_par[j]=tanh_mag*tanhf(received_signal_par[j]/tanh_mag);
                }
                if (i==0) pred_h=received_signal_par;
                float surprise=0.0f; //surprise=squared sum of error
                for (int j=0;j<d;j++){
                    nerr[i*d+j]=received_signal_par[j]-h[i*d+j];
                    surprise+=nerr[i*d+j]*nerr[i*d+j];
                }
                //error covariance matrix
                for (int j=0;j<d;j++){
                    for (int k=0;k<d;k++){
                        err_cov[i*d*d+j*d+k]+=dt*(nerr[i*d+j]*nerr[i*d+k]-err_cov[i*d*d+j*d+k]);
                    }
                }
                //step 2: aggregate error from children
                fill(all(emitted_signal_chl), 0.0f);
                for (auto chl:adj[i]){
                    matvec_transpose(a, err, chl_signal, d, r, chl, chl);
                    for (int j=0;j<r;j++){
                        emitted_signal_chl[j]+=tanh_mag*tanhf(chl_signal[j]/tanh_mag)/adj[i].size();
                    }
                }
                matvec(b, emitted_signal_chl, received_signal_chl, d, r, i, 0);
                for (int j=0;j<d;j++){
                    received_signal_chl[j]=tanh_mag*tanhf(received_signal_chl[j]/tanh_mag);
                }
                for (int j=0;j<d;j++){
                    nerr_diff[i*d+j]=received_signal_chl[j]-err[i*d+j];
                    surprise+=nerr_diff[i*d+j]*nerr_diff[i*d+j];
                }
                //move h
                for (int j=0;j<d;j++){
                    if (input[i]) continue;
                    dh[i*d+j]=dt*fast_learn*(nerr[i*d+j]-received_signal_chl[j]);
                    nh[i*d+j]=tanh_mag*tanhf((nh[i*d+j]+dh[i*d+j])/tanh_mag);
                }
                nu[i]+=dt*((surprise/(2*d)+mag(a, i*d*r, d*r)/(d*r)+mag(b, i*d*r, d*r)/(d*r))-u[i]);
                //move a
                delta_rule(a, emitted_signal_par, nerr, slow_learn, da, d, r, i);
                //move b (hypothesis here - align the errors, invert A=sort of "unify/harmonize" the whole network. success spreads, uncertainty scaling makes failures spread much less)
                delta_rule(b, emitted_signal_chl, nerr_diff, slow_learn, db, d, r, i);
            }
            swap(h, nh);
            swap(err, nerr);
            swap(u, nu);
            swap(err_diff, nerr_diff);
            int tempn=n;
            for (int i=0;i<tempn;i++){
                if (!input[i] && u[i]<omega && u[i]>tau){
                    mitosis(i);
                }
            }
            if (ticks>0 && ticks%cticks==0) cleanup();
            ticks++;
        }
        void update(int k){
            for (int i=0;i<k;i++) forward();
        }
        void step(){
            update(30);
        }
        //predict not necessary if something points into 0
        // vector<float> predict(){ //assumes input node is 0
        //     matmat_transpose(b, a, scratch, d, r, d, 0, 1);
        //     vector<float> scratch_copy=scratch;
        //     for (int i=0;i<d;i++) scratch_copy[i*d+i]+=dna[1].raw_signal_par;
        //     vector<float> target(d,0); for (int i=0;i<d;i++) target[i]=atanhf(h[1*d+i]/tanh_mag)*tanh_mag;
        //     vector<float> s(d);
        //     float eps=1e-5f;
        //     int rank; int info; float opt_work; int opt_iwork; int lwork_q=-1; int nrhs=1;
        //     sgelsd_(&d, &d, &nrhs, scratch_copy.data(), &d, target.data(), &d, s.data(), &eps, &rank, &opt_work, &lwork_q, &opt_iwork, &info);
        //     int lwork=(int)opt_work;
        //     vector<float> work(lwork);
        //     vector<int> iwork(opt_iwork);
        //     sgelsd_(&d, &d, &nrhs, scratch_copy.data(), &d, target.data(), &d, s.data(), &eps, &rank, work.data(), &lwork, iwork.data(), &info);
        //     if (info!=0) cout<<"error in predict (sgelsd_), info: "<<info<<'\n';
        //     //here we just ignore the nonlinearity on the inside, it's close enough to linear and can be changed
        //     for (int i=0;i<d;i++) target[i]=target[i]*(1.0f+u[0]);
        //     return target;
        // }
    };
vector<bool> un_input(2, false); //input mask
void update_data(float &x, float &y, float &z, int i, string tp) {
    //some of these were written with ai
    if (tp=="lorenz"){
        //LORENZ
        const float sigma0 = 10.0f;
        const float rho0   = 28.0f;
        const float beta0  = 8.0f / 3.0f;
        float dx = sigma0 * (y - x);
        float dy = x * (rho0 - z) - y;
        float dz = x * y - beta0 * z;
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;
    }
    if (tp=="sin"){
        float t = i * dt;
        float freq = 0.2f;
        x = 2.0f * sinf(t * freq);
        y = 2.0f * sinf(t * freq + 1.57f);
        z = 2.0f * sinf(t * freq * 1.5f);
    }
    if (tp=="brownian"){
        //BROWNIAN
        x += gaussian_noise(0.0f, 20.0f * sqrtf(dt)); 
        y += gaussian_noise(0.0f, 27.0f * sqrtf(dt));
        z += gaussian_noise(0.0f, 25.0f * sqrtf(dt));
        x = max(-20.0f, min(20.0f, x));
        y = max(-27.0f, min(27.0f, y));
        z = max(0.0f, min(50.0f, z));
    }
    if (tp=="rossler"){
        //ROSSLER
        const float a = 0.2f, b = 0.2f, c = 5.7f;
        float dx = -y - z;
        float dy = x + a * y;
        float dz = b + z * (x - c);
        x += dx * dt * 1.5f;
        y += dy * dt * 1.5f;
        z += dz * dt * 1.5f;
    }
    if (tp=="fourier"){
        //FOURIER
        float t = i * dt;
        x=0.7f*sinf(t*1.3f)+0.7f*cosf(t*2.7f)+0.6f*sinf(t*4.1f);
        y=0.7f*sinf(t*1.9f)+0.7f*cosf(t*3.3f)+0.6f*sinf(t*5.7f);
        z=0.7f*sinf(t*1.1f)+0.7f*cosf(t * 2.1f)+0.6f*sinf(t*3.9f);
    }
    if (tp=="fourier_rev"){
        //FOURIER, REVERSED
        float t = (99999-i) * dt;
        x=0.7f*sinf(t*1.3f)+0.7f*cosf(t*2.7f)+0.6f*sinf(t*4.1f);
        y=0.7f*sinf(t*1.9f)+0.7f*cosf(t*3.3f)+0.6f*sinf(t*5.7f);
        z=0.7f*sinf(t*1.1f)+0.7f*cosf(t * 2.1f)+0.6f*sinf(t*3.9f);
    }
    if (tp=="o-u"){
        //O-U
        float pull_strength = 2.0f;
        float noise_mag = 15.0f;
        x += -pull_strength * x * dt + gaussian_noise(0.0f, noise_mag * sqrtf(dt));
        y += -pull_strength * y * dt + gaussian_noise(0.0f, noise_mag * sqrtf(dt));
        z += -pull_strength * (z - 25.0f) * dt + gaussian_noise(0.0f, noise_mag * sqrtf(dt));
        x = max(-20.0f, min(20.0f, x));
        y = max(-27.0f, min(27.0f, y));
        z = max(0.0f, min(50.0f, z));
    }
    if (tp=="constant"){
        x=2.0f;
        y=-1.0f;
        z=3.0f;
    }
}
vector<int> input_nodes{0};
void run_exp(string curtp, int tot_num, int eval_start, int eval_end, bool do_learning) {
    cout<<"RUNNING EXPERIMENT "<<curtp<<endl;
    int lived=0;
    int died=0;
    int avgdeath=0;
    float f_mse=0.0f;
    float f_skill=0.0f;
    float f_cos=0.0f;
    float f_amp=0.0f;
    for (int num=1;num<=tot_num;num++){
        float curx=0.05f, cury=0.1f, curz=-0.025f;
        lupus sextus(un_input, 100);
        sextus.reset();
        if (!do_learning) sextus.slow_learn=0.0f;
        float avg_mse=0.0f;
        float avg_skill=0.0f;
        float avg_cos=0.0f;
        float avg_amp=0.0f;
        for (int i=0;i<100000;i++){
            //if (i==25000) sextus.mitosis(1);
            //cout<<"STEP NUMBER: "<<i+1<<endl;
            update_data(curx, cury, curz, i, curtp);
            //cout<<"POS AT: "<<curx<<", "<<cury<<", "<<curz<<endl;
            for (auto xx:input_nodes){
                sextus.h[xx*sextus.d+0]=curx/5.0f;
                sextus.h[xx*sextus.d+1]=cury/5.0f;
                sextus.h[xx*sextus.d+2]=curz/5.0f;
            }
            vector<float> pred=sextus.pred_h;
            
            // if (i%10000==0){
            //     cout<<"TICK: "<<i<<'\n';
            //     cout<<"[BEFORE STEP]: \n";
            //     cout<<"N: "<<sextus.n<<'\n';
            //     cout<<"NODE 0 U: "<<sextus.u[0]<<'\n';
            //     cout<<"NODE 1 U: "<<sextus.u[1]<<'\n';
            //     pred=sextus.pred_h;
            //     cout<<"prediction=("<<pred[0]<<", "<<pred[1]<<", "<<pred[2]<<", "<<pred[3]<<")\n";
            //     cout<<"NODE 1 H: ("<<sextus.h[1*sextus.d+0]<<", "<<sextus.h[1*sextus.d+1]<<", "<<sextus.h[1*sextus.d+2]<<", "<<sextus.h[1*sextus.d+3]<<")\n";
            // }
            // if (i%20000==0){
            //     cout<<"TICK "<<i<<endl;
            //     cout<<"N: "<<sextus.n<<endl;
            //     for (int j=0;j<sextus.n;j++){
            //         cout<<"NODE "<<j+1
            //         <<", A_MAG: "<<mag(sextus.a, j*sextus.d*sextus.r, sextus.d*sextus.r)
            //         <<", B_MAG: "<<mag(sextus.b, j*sextus.d*sextus.r, sextus.d*sextus.r)
            //         <<", U: "<<sextus.u[j]
            //         <<", energy: "<<sextus.e[j]
            //         <<", EVS: ("<<sextus.E[j].val<<", "<<sextus.V[j].val<<", "<<sextus.S[j].val<<")"
            //         <<", STRESS: "<<sextus.stress[j]
            //         <<endl;
            //     }
            // }
            sextus.step();
            vector<float> tgt={curx/5.0f, cury/5.0f, curz/5.0f, 0.0f};
            float mse=0.0f;
            float dot=0.0f;
            float pred_norm=mag(pred, 0, sextus.d);
            float tgt_norm=mag(tgt, 0, sextus.d);
            for (int j=0;j<sextus.d;j++){
                float pred_err=tgt[j]-pred[j];
                mse+=(pred_err*pred_err)/sextus.d;
                dot+=pred[j]*tgt[j];
            }
            float cos_sim=dot/sqrtf(max(pred_norm*tgt_norm, 1e-7f));
            float amp=sqrtf(pred_norm/tgt_norm);
            float skill=1-mse/(tgt_norm/sextus.d);
            if (eval_start<=i && i<=eval_end){
                avg_mse+=mse/(eval_end-eval_start+1);
                avg_skill+=skill/(eval_end-eval_start+1);
                avg_cos+=cos_sim/(eval_end-eval_start+1);
                avg_amp+=amp/(eval_end-eval_start+1);
            }
            // if (i%10000==0){
            //     cout<<"[PREDICTION STATS]: \n";
            //     cout<<"MSE: "<<mse
            //     <<"\nSKILL: "<<skill
            //     <<"\nCOS SIMILARITY: "<<cos_sim
            //     <<"\nAMP: "<<amp<<"\n";
            // }
            int tot=0;
            for (int j=0;j<sextus.n;j++){
                if (!sextus.input[j] && sextus.u[j]>sextus.omega) tot++;
            }
            if (tot==sextus.n-input_nodes.size()) {
                cout<<"DEATH "<<num<<": "<<i<<endl;
                died++;
                avgdeath+=i;
                break;
            }
            if (i==99999) {
                lived++;
                avgdeath+=100000;
            }
        }
        cout<<"RUN "<<num<<" AVERAGE STATS FROM "<<eval_start<<" TO "<<eval_end<<": \n";
        cout<<"MSE: "<<avg_mse
        <<"\nSKILL: "<<avg_skill
        <<"\nCOS SIMILARITY: "<<avg_cos
        <<"\nAMP: "<<avg_amp<<"\n";
        f_mse+=avg_mse/tot_num;
        f_skill+=avg_skill/tot_num;
        f_cos+=avg_cos/tot_num;
        f_amp+=avg_amp/tot_num;
    }
    cout<<"LIVED: "<<lived<<endl;
    cout<<"DIED: "<<died<<endl;
    cout<<"AVERAGE SPAN: "<<avgdeath/tot_num<<endl;
    cout<<"OVERALL AVERAGE STATS FROM "<<eval_start<<" TO "<<eval_end<<": \n";
    cout<<"MSE: "<<f_mse
    <<"\nSKILL: "<<f_skill
    <<"\nCOS SIMILARITY: "<<f_cos
    <<"\nAMP: "<<f_amp<<"\n";
}
void run_special(string tp1, string tp2, int tot_num, int eval_start, int eval_end) {
    int lived=0;
    int died=0;
    int avgdeath=0;
    float f_mse1=0.0f;
    float f_skill1=0.0f;
    float f_cos1=0.0f;
    float f_amp1=0.0f;
    float f_mse2=0.0f;
    float f_skill2=0.0f;
    float f_cos2=0.0f;
    float f_amp2=0.0f;
    for (int num=1;num<=tot_num;num++){
        float curx=0.05f, cury=0.1f, curz=-0.025f;
        lupus sextus(un_input, 100);
        sextus.reset();
        float avg_mse1=0.0f;
        float avg_skill1=0.0f;
        float avg_cos1=0.0f;
        float avg_amp1=0.0f;
        float avg_mse2=0.0f;
        float avg_skill2=0.0f;
        float avg_cos2=0.0f;
        float avg_amp2=0.0f;
        for (int i=0;i<100000*2;i++){
            //if (i==25000) sextus.mitosis(1);
            //cout<<"STEP NUMBER: "<<i+1<<endl;
            if (i>=100000){
                sextus.slow_learn=0.0f;
                update_data(curx, cury, curz, i-100000, tp2);
            } else{
                update_data(curx, cury, curz, i, tp1);
            }
            //cout<<"POS AT: "<<curx<<", "<<cury<<", "<<curz<<endl;
            for (auto xx:input_nodes){
                sextus.h[xx*sextus.d+0]=curx/5.0f;
                sextus.h[xx*sextus.d+1]=cury/5.0f;
                sextus.h[xx*sextus.d+2]=curz/5.0f;
            }
            vector<float> pred=sextus.pred_h;
            
            // if (i%10000==0){
            //     cout<<"TICK: "<<i<<'\n';
            //     cout<<"[BEFORE STEP]: \n";
            //     cout<<"N: "<<sextus.n<<'\n';
            //     cout<<"NODE 0 U: "<<sextus.u[0]<<'\n';
            //     cout<<"NODE 1 U: "<<sextus.u[1]<<'\n';
            //     pred=sextus.pred_h;
            //     cout<<"prediction=("<<pred[0]<<", "<<pred[1]<<", "<<pred[2]<<", "<<pred[3]<<")\n";
            //     cout<<"NODE 1 H: ("<<sextus.h[1*sextus.d+0]<<", "<<sextus.h[1*sextus.d+1]<<", "<<sextus.h[1*sextus.d+2]<<", "<<sextus.h[1*sextus.d+3]<<")\n";
            // }
            // if (i%20000==0){
            //     cout<<"TICK "<<i<<endl;
            //     cout<<"N: "<<sextus.n<<endl;
            //     for (int j=0;j<sextus.n;j++){
            //         cout<<"NODE "<<j+1
            //         <<", A_MAG: "<<mag(sextus.a, j*sextus.d*sextus.r, sextus.d*sextus.r)
            //         <<", B_MAG: "<<mag(sextus.b, j*sextus.d*sextus.r, sextus.d*sextus.r)
            //         <<", U: "<<sextus.u[j]
            //         <<", energy: "<<sextus.e[j]
            //         <<", EVS: ("<<sextus.E[j].val<<", "<<sextus.V[j].val<<", "<<sextus.S[j].val<<")"
            //         <<", STRESS: "<<sextus.stress[j]
            //         <<endl;
            //     }
            // }
            sextus.step();
            vector<float> tgt={curx/5.0f, cury/5.0f, curz/5.0f, 0.0f};
            float mse=0.0f;
            float dot=0.0f;
            float pred_norm=mag(pred, 0, sextus.d);
            float tgt_norm=mag(tgt, 0, sextus.d);
            for (int j=0;j<sextus.d;j++){
                float pred_err=tgt[j]-pred[j];
                mse+=(pred_err*pred_err)/sextus.d;
                dot+=pred[j]*tgt[j];
            }
            float cos_sim=dot/sqrtf(max(pred_norm*tgt_norm, 1e-7f));
            float amp=sqrtf(pred_norm/tgt_norm);
            float skill=1-mse/(tgt_norm/sextus.d);
            if (eval_start<=i && i<=eval_end){
                avg_mse1+=mse/(eval_end-eval_start+1);
                avg_skill1+=skill/(eval_end-eval_start+1);
                avg_cos1+=cos_sim/(eval_end-eval_start+1);
                avg_amp1+=amp/(eval_end-eval_start+1);
            }
            if (eval_start<=i-100000 && i-100000<=eval_end){
                avg_mse2+=mse/(eval_end-eval_start+1);
                avg_skill2+=skill/(eval_end-eval_start+1);
                avg_cos2+=cos_sim/(eval_end-eval_start+1);
                avg_amp2+=amp/(eval_end-eval_start+1);
            }
            // if (i%10000==0){
            //     cout<<"[PREDICTION STATS]: \n";
            //     cout<<"MSE: "<<mse
            //     <<"\nSKILL: "<<skill
            //     <<"\nCOS SIMILARITY: "<<cos_sim
            //     <<"\nAMP: "<<amp<<"\n";
            // }
            int tot=0;
            for (int j=0;j<sextus.n;j++){
                if (!sextus.input[j] && sextus.u[j]>sextus.omega) tot++;
            }
            if (tot==sextus.n-input_nodes.size()) {
                cout<<"DEATH "<<num<<": "<<i<<endl;
                died++;
                avgdeath+=i;
                break;
            }
            if (i==99999) {
                lived++;
                avgdeath+=100000;
            }
        }
        cout<<"RUN "<<num<<" AVERAGE STATS FROM "<<eval_start<<" TO "<<eval_end<<": \n";
        cout<<"MSE 1: "<<avg_mse1
        <<"\nSKILL 1: "<<avg_skill1
        <<"\nCOS SIMILARITY 1: "<<avg_cos1
        <<"\nAMP 1: "<<avg_amp1<<"\n";
        cout<<"MSE 2: "<<avg_mse2
        <<"\nSKILL 2: "<<avg_skill2
        <<"\nCOS SIMILARITY 2: "<<avg_cos2
        <<"\nAMP 2: "<<avg_amp2<<"\n";
        f_mse1+=avg_mse1/tot_num;
        f_skill1+=avg_skill1/tot_num;
        f_cos1+=avg_cos1/tot_num;
        f_amp1+=avg_amp1/tot_num;
        f_mse2+=avg_mse2/tot_num;
        f_skill2+=avg_skill2/tot_num;
        f_cos2+=avg_cos2/tot_num;
        f_amp2+=avg_amp2/tot_num;
    }
    cout<<"LIVED: "<<lived<<endl;
    cout<<"DIED: "<<died<<endl;
    cout<<"AVERAGE SPAN: "<<avgdeath/tot_num<<endl;
    cout<<"OVERALL AVERAGE STATS FROM "<<eval_start<<" TO "<<eval_end<<": \n";
    cout<<"MSE 1: "<<f_mse1
    <<"\nSKILL1: "<<f_skill1
    <<"\nCOS SIMILARITY 1: "<<f_cos1
    <<"\nAMP 1: "<<f_amp1<<"\n";
    cout<<"MSE 2: "<<f_mse2
    <<"\nSKILL2: "<<f_skill2
    <<"\nCOS SIMILARITY 2: "<<f_cos2
    <<"\nAMP 2: "<<f_amp2<<"\n";
}

int main(){
    for (auto xx:input_nodes){
        un_input[xx]=true;
    }
    vector<string> experiments{"lorenz", "sin", "brownian", "rossler", "fourier", "o-u", "constant"};
    //experiments={"fourier"};
    //for (auto curtp:experiments){
    //    run_exp(curtp, 20, 50000, 100000);
    //}
    run_special("fourier", "fourier_rev", 20, 50000, 100000);
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/
