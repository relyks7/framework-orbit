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
float SENTINEL=-0.1225711;
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
    //0 is input, 1 is cortex, 2 is motor
    public:
        vector<unordered_set<int>> adj;
        vector<bool> input; vector<bool> motor;
        vector<unordered_set<int>> radj;
        vector<float> h; //internal state
        vector<float> dh;
        vector<float> err; //error
        vector<float> err_diff;
        vector<float> err_cov; //error matrix (uncentered second moment, not covariance)
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
        vector<float> setpoints;
        vector<bool> o_input;
        vector<bool> o_motor;
        vector<float> o_setpoints;
        int ticks=0;
        int cticks;
        int n=3;
        int d=2;
        int r=8;
        float fast_learn;
        float slow_learn;
        float tau;
        float omega;
        lupus(const vector<bool>& uinput, const vector<bool>& umotor, int icticks, vector<float> isetpoints){
            adj.assign(n,unordered_set<int>{}); radj.assign(n,unordered_set<int>{}); input=uinput; motor=umotor;
            o_input=uinput;
            o_motor=umotor;
            o_setpoints=isetpoints;
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
            setpoints=isetpoints;
            fast_learn=1.0f; slow_learn=0.02f;
            tau=100.5f; omega=0.75f; //mitosis is disabled
        }
        void cleanup(){ //apoptosis, run every k ticks
            vector<bool> alive(n,true);
            vector<int> new_index(n,-1);
            int cnt=0;
            for (int i=0;i<n;i++){
                if (!input[i] && !motor[i] && u[i]>omega){
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
            vector<float> nsetpoints{};
            vector<bool> ninput;
            vector<bool> nmotor;
            for (int i=0;i<n;i++){
                if (alive[i]){
                    ninput.push_back(input[i]);
                    nmotor.push_back(motor[i]);
                    nu.push_back(u[i]);
                    for (int j=0;j<d;j++){
                        nh.push_back(h[i*d+j]);
                        nsetpoints.push_back(setpoints[i*d+j]);
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
            swap(setpoints, nsetpoints);
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
            swap(motor, nmotor);
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
                setpoints.push_back(setpoints[i*d+j]);
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
                setpoints.push_back(setpoints[i*d+j]);
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
            motor.push_back(false);
            motor.push_back(false);
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
            n=3;
            adj.assign(n,unordered_set<int>{}); radj.assign(n,unordered_set<int>{});
            input=o_input;
            motor=o_motor;
            setpoints=o_setpoints;
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

            adj[1].insert(2);
            radj[2].insert(1);
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
                    if (setpoints[i*d+j]==SENTINEL) nerr[i*d+j]=received_signal_par[j]-h[i*d+j]; 
                    else nerr[i*d+j]=setpoints[i*d+j]-h[i*d+j];
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
                    dh[i*d+j]=dt*tanh_mag*tanhf(fast_learn*(nerr[i*d+j]-received_signal_chl[j])/tanh_mag);
                    nh[i*d+j]=nh[i*d+j]+dh[i*d+j];
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
                if (!input[i] && !motor[i] && u[i]<omega && u[i]>tau){
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
        vector<float> get_motor(){ //general to handle more motor nodes, currently only 1
            vector<float> ret(d,0);
            for (int i=0;i<n;i++){
                if (!motor[i]) continue;
                for (int j=0;j<d;j++){
                    ret[j]+=h[i*d+j];
                }
            }
            return ret;
        }
};
vector<int> input_nodes{0};
vector<int> motor_nodes{2};
vector<bool> un_input(3, false); //input mask
vector<bool> un_motor(3, false); //motor mask
vector<float> un_setpoints(3*2, SENTINEL); //setpoint priors
lupus linetest(float prox, int ticks, lupus s, float goal){ //proximity for goal
    // float goalx=disf(rng)*2-1.0f;
    float goalx=goal;
    float curx=0.0f;
    for (int i=0;i<ticks;i++){
        s.setpoints[1]=goalx;
        s.h[0]=curx;
        s.h[1]=curx;
        s.step();
        float movex=tanhf(s.get_motor()[0]);
        curx+=movex*dt;
        curx=max(-1.0f, min(1.0f, curx));
        // if (abs(goalx-curx)<prox) cout<<"hit goal\n";
        // while(abs(goalx-curx)<prox) {
        //     goalx=disf(rng)*2-1.0f;
        // }
        if (i%100==0) cout<<"curx: "<<curx<<"\ngoal: "<<goalx<<'\n';
    }
    return s;
}
int main(){
    for (auto xx:input_nodes){
        un_input[xx]=true;
    }
    for (auto xx:motor_nodes){
        un_motor[xx]=true;
    }
    lupus sextus(un_input, un_motor, 100, un_setpoints);
    sextus.reset();
    cout<<"early half run:\n";
    sextus=linetest(0.01f, 40000, sextus, 0.4f);
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/
