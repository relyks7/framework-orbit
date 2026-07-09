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
float starting_energy=50.0f;
float delta_clamp=0.25f;
float phi=1.618033988749895;
float tanh_mag=10.0f;
float mitosis_threshold=0.85; //tau
float death_energy=40.0f; //omega
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
    return cblas_sdot(size, a.data()+start_idx, 1, a.data()+start_idx, 1)/size;
}
void delta_rule(vector<float> &a, const vector<float>&x, const vector<float>&err, float lr, vector<float>& da, float a_dec, int n, int m, int idx){
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            da[idx*m*n+i*m+j]=max(-delta_clamp, min(delta_clamp, -lr*dt*err[idx*n+i]*x[j]));
            a[idx*m*n+i*m+j]+=-dt*a_dec*a[idx*m*n+i*m+j]+da[idx*m*n+i*m+j];
        }
    }
}
struct genes{ //"all roads lead to darwinism"
    float dh_chl_contrib;
    float dh_par_contrib;
    float fast_adaptation_rate;
    float slow_adaptation_learning_rate_a;
    float slow_adaptation_learning_rate_b;
    float h_decay;
    float e_decay;
    float cost_of_thought;
    float cost_of_plasticity;
    float cost_of_complexity;
    float correct_income;
    float a_decay;
    float b_decay;
    float stress_decay;
    float raw_signal_par;
    float raw_signal_chl;
    float stress_grow;
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
class trait{ //decay=plasticity
    public:
        float val;
        float target_val;
        float plasticity;
        float decay;
        float resp;
        float noise;
        void step(float feeder_val){
            feeder_val=max(0.0f, min(1.0f, feeder_val));
            target_val=target_val+dt*(plasticity*feeder_val-decay*target_val)+sqrtf(dt)*gaussian_noise(0.0f, noise);
            target_val=max(0.01f,min(0.99f,target_val));
            val+=dt*resp*(target_val-val);
            val=max(0.01f,min(0.99f,val));
        }
};
class lupus{
    //0 is prior, 1 is eyes, 2 is motor
    public:
        vector<unordered_set<int>> adj; //initialize with smallworld graph
        vector<bool> input;
        vector<unordered_set<int>> radj;
        vector<float> h; //internal state
        vector<float> dh;
        vector<float> e; //energy, initially high
        vector<float> err; //error
        vector<float> err_diff;
        vector<float> err_cov; //error covariance matrix
        vector<float> u; //uncertainty
        vector<float> a; //receptors
        vector<float> da;
        vector<float> b; //emitters
        vector<float> db;
        vector<float> stress;
        //scratch
        vector<float> par_signal;
        vector<float> emitted_signal_par;
        vector<float> received_signal_par;
        vector<float> nu, nh, nerr, nerr_diff;
        vector<float> emitted_signal_chl;
        vector<float> chl_signal;
        vector<float> received_signal_chl;
        vector<float> scratch;
        vector<genes> dna;
        vector<trait> E;
        vector<trait> V;
        vector<trait> S;
        trait iE;
        trait iV;
        trait iS;
        int ticks=0;
        int cticks;
        int n=2;
        int d=4;
        int r=32;
        lupus(const trait& initE, const trait& initV, const trait& initS, const vector<bool>& uinput, int icticks){
            adj.assign(n,unordered_set<int>{}); radj.assign(n,unordered_set<int>{}); dna.assign(n,genes()); input=uinput;
            h.assign(n*d, 0.0f);
            dh.assign(n*d, 0.0f);
            e.assign(n, starting_energy);
            err.assign(n*d, 0.0f);
            err_cov.assign(n*d*d, 0.0f);
            err_diff.assign(n*d, 0.0f);
            u.assign(n, 0.0f);
            a.assign(n*d*r, 0.0f);
            da.assign(n*d*r, 0.0f);
            b.assign(n*d*r, 0.0f);
            db.assign(n*d*r, 0.0f);
            stress.assign(n, 0.0f);
            par_signal.assign(r, 0.0f);
            emitted_signal_par.assign(r, 0.0f);
            received_signal_par.assign(d, 0.0f);
            chl_signal.assign(r, 0.0f);
            emitted_signal_chl.assign(r, 0.0f);
            received_signal_chl.assign(d, 0.0f);
            scratch.assign(d*d, 0.0f);
            cticks=icticks;
            E.assign(n, initE);
            V.assign(n, initV);
            S.assign(n, initS);
            iE=initE;
            iV=initV;
            iS=initS;
        }
        void cleanup(){ //apoptosis, run every k ticks
            vector<bool> alive(n,true);
            vector<int> new_index(n,-1);
            int cnt=0;
            for (int i=0;i<n;i++){
                if (e[i]<=death_energy){
                    alive[i]=false; 
                    cout<<"NODE "<<i<<" DEAD IN CLEANUP"<<endl;
                    continue;
                }
                new_index[i]=cnt++;
            }
            nh.clear();
            vector<float> ndh{};
            vector<float> ne{};
            nerr.clear();
            nerr_diff.clear();
            nu.clear();
            vector<float> na{};
            vector<float> nda{};
            vector<float> nb{};
            vector<float> ndb{};
            vector<float> nstress{};
            vector<genes> ndna{};
            vector<float> nerr_cov{};
            vector<trait> nE;
            vector<trait> nV;
            vector<trait> nS;
            vector<bool> ninput;
            for (int i=0;i<n;i++){
                if (alive[i]){
                    ninput.push_back(input[i]);
                    nE.push_back(E[i]);
                    nV.push_back(V[i]);
                    nS.push_back(S[i]);
                    ne.push_back(e[i]);
                    nu.push_back(u[i]);
                    nstress.push_back(stress[i]);
                    ndna.push_back(dna[i]);
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
            swap(e, ne);
            swap(err, nerr);
            swap(err_diff, nerr_diff);
            swap(u, nu);
            swap(a, na);
            swap(da, nda);
            swap(b, nb);
            swap(db, ndb);
            swap(stress, nstress);
            swap(dna, ndna);
            swap(err_cov, nerr_cov);
            swap(E, nE);
            swap(V, nV);
            swap(S, nS);
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
        void mitosis (int i){ //no, kris did not mitose
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
                cout<<"eigendecomposition error, code: "<<info<<endl;
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
                    a.push_back(a[i*d*r+j*r+k]);
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
            e.push_back(e[i]);
            e.push_back(e[i]);
            stress.push_back(0.0f);
            stress.push_back(0.0f);
            stress[i]=0;
            E.push_back(E[i]);
            V.push_back(V[i]);
            S.push_back(S[i]);
            E.push_back(E[i]);
            V.push_back(V[i]);
            S.push_back(S[i]);
            dna.push_back(dna[i]);
            dna.push_back(dna[i]);
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
        void set_dna(){
            for (int i=0;i<n;i++){
                dna[i].dh_chl_contrib=0.6f;
                dna[i].dh_par_contrib=1.0f;
                dna[i].cost_of_complexity=0.5f;
                dna[i].cost_of_thought=0.1f;
                dna[i].raw_signal_par=0.3f;
                dna[i].raw_signal_chl=0.3f;
                dna[i].e_decay=0.002f;
                dna[i].correct_income=0.75f;
                dna[i].stress_grow=0.05f;
                dna[i].cost_of_plasticity=1.0f*(1.0f-V[i].val*(1.0f-S[i].val));
                dna[i].a_decay=0.005f*V[i].val*(1.0f-S[i].val);
                dna[i].b_decay=0.005f*V[i].val*(1.0f-S[i].val);
                dna[i].fast_adaptation_rate=10.0f*E[i].val*(0.8f*V[i].val+0.2f*(1-S[i].val));
                dna[i].slow_adaptation_learning_rate_a=0.025f*V[i].val*(0.8f*E[i].val+0.2f*(1-S[i].val));
                dna[i].slow_adaptation_learning_rate_b=0.025f*V[i].val*(0.8f*E[i].val+0.2f*(1-S[i].val));
                dna[i].h_decay=0.005f*V[i].val*(1-S[i].val);
                dna[i].stress_decay=0.001f+0.004f*S[i].val;
            }
        }
        void reset(){
            n=2;
            adj.assign(n,unordered_set<int>{}); radj.assign(n,unordered_set<int>{}); dna.assign(n,genes());
            if (input.size()>n){
                input.erase(input.begin()+n, input.end());
            }
            h.assign(n*d, 0.0f);
            dh.assign(n*d, 0.0f);
            e.assign(n, starting_energy);
            err.assign(n*d, 0.0f);
            err_cov.assign(n*d*d, 0.0f);
            err_diff.assign(n*d, 0.0f);
            u.assign(n, 0.0f);
            a.assign(n*d*r, 0.0f);
            da.assign(n*d*r, 0.0f);
            b.assign(n*d*r, 0.0f);
            db.assign(n*d*r, 0.0f);
            stress.assign(n, 0.0f);
            par_signal.assign(r, 0.0f);
            emitted_signal_par.assign(r, 0.0f);
            received_signal_par.assign(d, 0.0f);
            chl_signal.assign(r, 0.0f);
            emitted_signal_chl.assign(r, 0.0f);
            received_signal_chl.assign(d, 0.0f);
            scratch.assign(d*d, 0.0f);
            ticks=0;
            E.assign(n, iE);
            V.assign(n, iV);
            S.assign(n, iS);
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
            radj[1].insert(0);
            set_dna();
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
                        emitted_signal_par[j]+=tanh_mag*tanhf((1.0f/(1.0f+u[par])*par_signal[j])/tanh_mag);
                    }
                }
                matvec(a, emitted_signal_par, received_signal_par ,d, r, i, 0);
                for (auto par:radj[i]){
                    for (int j=0;j<d;j++){
                        received_signal_par[j]+=dna[i].raw_signal_par*h[par*d+j]/(1.0f+u[par]);
                    }
                }
                for (int j=0;j<d;j++){
                    received_signal_par[j]=tanh_mag*tanhf(received_signal_par[j]/tanh_mag);
                }
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
                        emitted_signal_chl[j]+=tanh_mag*tanhf((u[chl]/(1.0f+u[chl])*chl_signal[j])/tanh_mag);
                    }
                }
                matvec(b, emitted_signal_chl, received_signal_chl, d, r, i, 0);
                for (auto chl:adj[i]){
                    for (int j=0;j<d;j++){
                        received_signal_chl[j]+=dna[i].raw_signal_chl*err[chl*d+j]*u[chl]/(1.0f+u[chl]);
                    }
                }
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
                    float fast_adapt=dna[i].fast_adaptation_rate*(dna[i].dh_par_contrib*nerr[i*d+j]-dna[i].dh_chl_contrib*received_signal_chl[j]);
                    dh[i*d+j]=dt*(fast_adapt-dna[i].h_decay*h[i*d+j]);
                    nh[i*d+j]=tanh_mag*tanhf((nh[i*d+j]+dh[i*d+j])/tanh_mag);
                }
                nu[i]+=dt*((surprise/(2*d))-u[i]); //uncertainty is a moving average of surprise
                //move a
                delta_rule(a, emitted_signal_par, nerr, dna[i].slow_adaptation_learning_rate_a, da, dna[i].a_decay, d, r, i);
                //move b (hypothesis here - align the errors, invert A=sort of "unify/harmonize" the whole network. success spreads, uncertainty scaling makes failures spread much less)
                delta_rule(b, emitted_signal_chl, nerr_diff, dna[i].slow_adaptation_learning_rate_b, db, dna[i].b_decay, d, r, i);
                //update energy
                if (input[i]){
                    //it's hard for input nodes to do well, and they provide the only signal source, so they should always be able to emit signal (scaled by e/(1+e) * 1/(1+u))
                    e[i]=1000.0f;
                    nu[i]=0.0f;
                    stress[i]=0.0f;
                } else{
                    e[i]+=dt*(dna[i].correct_income*(expf(-20.0f*surprise/(2*d)))-(dna[i].cost_of_thought*mag(dh, i*d, d)+dna[i].cost_of_complexity*(mag(a, i*d*r, d*r)+mag(b, i*d*r, d*r))+dna[i].cost_of_plasticity*(mag(da, i*d*r, d*r)+mag(db, i*d*r, d*r)))-dna[i].e_decay*e[i]);
                    e[i]=max(0.0f, e[i]);
                    nu[i]=min(nu[i],10.0f);
                    stress[i]+=dt*(dna[i].stress_grow*nu[i]-dna[i].stress_decay*stress[i]);
                }
            }
            for (int i=0;i<n;i++){
                /*
                E: Do I have energy and am confident?
                V: Is something unstable or uncertain? Do I need to change?
                S: How stable am I? Do I trust myself? How certain am I? 
                */
               float e_s=e[i]/(starting_energy+e[i]);
               float u_s=u[i]/(1.0f+u[i]);
               float conf=1.0f/(1.0f+u[i]);
               float mv_mag=mag(dh, i*d, d)+mag(da, i*d*r, d*r)+mag(db, i*d*r, d*r);
               float mv=mv_mag/(1.0f+mv_mag);
               float stb=1.0f/(1.0f+mv_mag);
               E[i].step(e_s*conf);
               V[i].step(mv*0.5f+u_s*0.5f);
               S[i].step(conf*stb);
            }
            set_dna();
            swap(h, nh);
            swap(err, nerr);
            swap(u, nu);
            swap(err_diff, nerr_diff);
            int tempn=n;
            for (int i=0;i<tempn;i++){
                if (!input[i] && stress[i]>mitosis_threshold && e[i]>death_energy){
                    cout<<"BEFORE MITOSIS, ORGANISM TICK: "<<ticks
                    <<", NODE: "<<i
                    <<", E: "<<e[i]
                    <<", STRESS: "<<stress[i]
                    <<", U: "<<u[i]
                    <<", EVS: ("<<E[i].val<<", "<<V[i].val<<", "<<S[i].val<<")"
                    <<endl;
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
            update(1);
            //NOTA BENE: STEP SET TO ONE UPDATE FOR NOW FOR TESTING PURPOSES. SUCCESS LIKELY IMPROVES IF THIS IS HIGHER.
        }
        vector<float> predict(){ //assumes input node is 0
            matmat_transpose(b, a, scratch, d, r, d, 0, 1);
            vector<float> scratch_copy=scratch;
            for (int i=0;i<d;i++) scratch_copy[i*d+i]+=dna[1].raw_signal_par;
            vector<float> target(d,0); for (int i=0;i<d;i++) target[i]=atanhf(h[1*d+i]/tanh_mag)*tanh_mag;
            vector<float> s(d);
            float eps=1e-5f;
            int rank; int info; float opt_work; int opt_iwork; int lwork_q=-1; int nrhs=1;
            sgelsd_(&d, &d, &nrhs, scratch_copy.data(), &d, target.data(), &d, s.data(), &eps, &rank, &opt_work, &lwork_q, &opt_iwork, &info);
            int lwork=(int)opt_work;
            vector<float> work(lwork);
            vector<int> iwork(opt_iwork);
            sgelsd_(&d, &d, &nrhs, scratch_copy.data(), &d, target.data(), &d, s.data(), &eps, &rank, work.data(), &lwork, iwork.data(), &info);
            if (info!=0) cout<<"error in predict (sgelsd_), info: "<<info<<'\n';
            //here we just ignore the nonlinearity on the inside, it's close enough to linear and can be changed
            for (int i=0;i<d;i++) target[i]=target[i]*(1.0f+u[0]);
            return target;
        }
    };
vector<bool> un_input(2, false); //input mask
void update_data(float &x, float &y, float &z, int i, string tp) {
    //NOTA BENE: MOST OF THESE TESTING THINGS WERE WRITTEN BY AI TO MAKE SURE THAT THEY HAD ROUGHLY THE SAME MAGNITUDE, AND ALSO JUST HOW THEY WORK IN THE FIRST PLACE - MEANT FOR A QUICK NULL HYPOTHESIS TEST AND A FEW ABLATION TESTS
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
        //SIN
        float time_now = i * dt; 
        float freq = 2.0f;
        x = 20.0f * sinf(time_now * freq); 
        y = 27.0f * sinf(time_now * freq + 1.57f);
        z = 25.0f * sinf(time_now * (freq * 1.5f)) + 25.0f;
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
        x = 8.0f * (sinf(t * 1.3f) + cosf(t * 2.7f) + sinf(t * 4.1f));
        y = 8.0f * (sinf(t * 1.9f) + cosf(t * 3.3f) + sinf(t * 5.7f));
        z = 8.0f * (sinf(t * 1.1f) + cosf(t * 2.1f) + sinf(t * 3.9f)) + 25.0f;
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
void run_exp(string curtp, int tot_num) {
    cout<<"RUNNING EXPERIMENT "<<curtp<<endl;
    int lived=0;
    int died=0;
    int avgdeath=0;
    for (int num=1;num<=tot_num;num++){
        float curx=0.05f, cury=0.1f, curz=-0.025f;
        trait iE{0.35f, 0.35f, 0.02f, 0.02f, 0.05f, 0.003f};
        trait iV{0.35f, 0.35f, 0.08f, 0.08f, 0.20f, 0.006f};
        trait iS{0.35f, 0.35f, 0.03f, 0.03f, 0.08f, 0.002f};
        lupus sextus(iE, iV, iS, un_input, 100);
        sextus.reset();
        for (int i=0;i<100000;i++){
            //cout<<"STEP NUMBER: "<<i+1<<endl;
            update_data(curx, cury, curz, i, curtp);
            //cout<<"POS AT: "<<curx<<", "<<cury<<", "<<curz<<endl;
            for (auto xx:input_nodes){
                sextus.h[xx*sextus.d+0]=curx/5.0f;
                sextus.h[xx*sextus.d+1]=cury/5.0f;
                sextus.h[xx*sextus.d+2]=curz/5.0f;
            }
            if (i%20000==0){
                cout<<"TICK "<<i<<endl;
                cout<<"N: "<<sextus.n<<endl;
                for (int j=0;j<sextus.n;j++){
                    cout<<"NODE "<<j+1
                    <<", A_MAG: "<<mag(sextus.a, j*sextus.d*sextus.r, sextus.d*sextus.r)
                    <<", B_MAG: "<<mag(sextus.b, j*sextus.d*sextus.r, sextus.d*sextus.r)
                    <<", U: "<<sextus.u[j]
                    <<", energy: "<<sextus.e[j]
                    <<", EVS: ("<<sextus.E[j].val<<", "<<sextus.V[j].val<<", "<<sextus.S[j].val<<")"
                    <<", STRESS: "<<sextus.stress[j]
                    <<endl;
                }
            }
            sextus.step();
            int tot=0;
            for (int j=0;j<sextus.n;j++){
                if (sextus.e[j]<death_energy) tot++;
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
    }
    cout<<"LIVED: "<<lived<<endl;
    cout<<"DIED: "<<died<<endl;
    cout<<"AVERAGE SPAN: "<<avgdeath/tot_num<<endl;
}
int main(){
    for (auto xx:input_nodes){
        un_input[xx]=true;
    }
    vector<string> experiments{"lorenz", "sin", "brownian", "rossler", "fourier", "o-u", "constant"};
    for (auto curtp:experiments){
        run_exp(curtp, 2);
    }
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/