#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <thread>
#include <iomanip>
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
int n=100;
int d=4;
int r=32;
int deg=8;
float starting_energy=20.0f;
float delta_clamp=1.0f;
float fr=40.0f; //food radius
float speed=8.0f;
float bound=200.0f;
float hunger_drain=0.01f;
float start_dist=30.0f;
float within_dist=0.2f; //tolerance for differentiating direcion
float phi=1.618033988749895;
float tanh_mag=10.0f;
void matvec(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int idx1, int idx2){
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*m), 1, 0.0f, c.data(), 1);
}
void matvec_transpose(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int idx1, int idx2){
    cblas_sgemv(CblasRowMajor, CblasTrans, n, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*n), 1, 0.0f, c.data(), 1);
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
struct genes{ //"all roads lead to darwinism"
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
class trait{ //preferably, decay=plasticity
    public:
        float val;
        float target_val;
        float decay;
        float v_decay;
        float voltage;
        float capacity;
        float noise;
        float plasticity;
        void step(float feeder_val){
            target_val=target_val+dt*(plasticity*feeder_val-decay*target_val+gaussian_noise(0.0f, noise));
            target_val=max(0.1f,min(0.9f,target_val));
            voltage+=dt*(abs(feeder_val)-voltage*(1-v_decay));
            if (voltage>capacity){
                voltage=0.0f;
                val=target_val;
            }
        }
};
class lupus{
    //0 is prior, 1 is eyes, 2 is motor
    public:
        vector<vector<int>> adj; //initialize with smallworld graph
        vector<bool> input;
        vector<vector<int>> radj;
        vector<float> h; //internal state
        vector<float> dh;
        vector<float> e; //energy, initially high
        vector<float> err; //error
        vector<float> u; //uncertainty
        vector<float> a; //receptors
        vector<float> da;
        vector<float> b; //emitters
        vector<float> db;
        //scratch
        vector<float> par_signal;
        vector<float> emitted_signal_par;
        vector<float> received_signal_par;
        vector<float> nu, nh, nerr;
        vector<float> err_diff;
        vector<float> emitted_signal_chl;
        vector<float> chl_signal;
        vector<float> received_signal_chl;
        vector<genes> dna;
        vector<trait> E;
        vector<trait> V;
        vector<trait> S;
        lupus(const genes& rdna, const trait& initE, const trait& initV, const trait& initS, const vector<bool>& uinput){
            adj.assign(n,vector<int>{}); radj.assign(n,vector<int>{}); dna.assign(n,rdna); input=uinput;
            h.assign(n*d, 0);
            dh.assign(n*d, 0);
            e.assign(n, starting_energy);
            err.assign(n*d, 0);
            err_diff.assign(n*d,0);
            u.assign(n,0.0f);
            a.assign(n*d*r, 0);
            da.assign(n*d*r, 0);
            b.assign(n*d*r, 0);
            db.assign(n*d*r, 0);
            par_signal.assign(r,0);
            emitted_signal_par.assign(r,0);
            received_signal_par.assign(d,0);
            chl_signal.assign(r,0);
            emitted_signal_chl.assign(r,0);
            received_signal_chl.assign(d,0);
            E.assign(n,initE);
            V.assign(n,initV);
            S.assign(n,initS);
        }
        void reset(float p){
            fill(all(h), 0.0f);
            fill(all(dh), 0.0f);
            fill(all(err), 0.0f);
            fill(all(u), 0.0f);
            fill(all(da), 0.0f);
            fill(all(db), 0.0f);
            fill(all(e), starting_energy);
            adj.assign(n,vector<int>{}); radj.assign(n,vector<int>{});
            generate_smallworld(p, adj, radj);
            for (int i=0;i<n;i++){
                for (int j=0;j<d;j++){
                    for (int k=0;k<r;k++){
                        a[i*d*r+j*r+k]=gaussian_noise(0.0f, 1.0f/sqrtf(r));
                        b[i*d*r+j*r+k]=gaussian_noise(0.0f, 1.0f/sqrtf(r));
                    }
                }
            }
        }
        void forward(){
            nu=u;
            nh=h;
            nerr=err;
            for (int i=0;i<n;i++){
                //step 1: aggregate belief from parents
                fill(all(emitted_signal_par), 0.0f);
                for (auto par:radj[i]){
                    matvec_transpose(b, h, par_signal, d, r, par, par);
                    for (int j=0;j<r;j++){
                        emitted_signal_par[j]+=tanh_mag*tanhf((e[par]/(1.0f+e[par]))*(1.0f/(1.0f+u[par])*par_signal[j])/tanh_mag);
                    }
                }
                matvec(a, emitted_signal_par, received_signal_par ,d, r, i, 0);
                for (int j=0;j<d;j++){
                    received_signal_par[j]=tanh_mag*tanhf(received_signal_par[j]/tanh_mag);
                }
                float surprise=0.0f; //surprise=squared sum of error
                for (int j=0;j<d;j++){
                    nerr[i*d+j]=received_signal_par[j]-h[i*d+j];
                    surprise+=nerr[i*d+j]*nerr[i*d+j];
                }
                //step 2: aggregate error from children
                fill(all(emitted_signal_chl), 0.0f);
                for (auto chl:adj[i]){
                    matvec_transpose(a, err, chl_signal, d, r, chl, chl);
                    for (int j=0;j<r;j++){
                        emitted_signal_chl[j]+=tanh_mag*tanhf((e[chl]/(1.0f+e[chl]))*(1.0f/(1.0f+u[chl])*chl_signal[j])/tanh_mag);
                    }
                }
                matvec(b, emitted_signal_chl, received_signal_chl, d, r, i, 0);
                for (int j=0;j<d;j++){
                    received_signal_chl[j]=tanh_mag*tanhf(received_signal_chl[j]/tanh_mag);
                }
                for (int j=0;j<d;j++){
                    err_diff[i*d+j]=received_signal_chl[j]-err[i*d+j];
                    surprise+=err_diff[i*d+j]*err_diff[i*d+j];
                }
                float sig_mag=mag(received_signal_par, 0, received_signal_par.size())+mag(received_signal_chl, 0, received_signal_chl.size()); //magnitude of received signal
                //move h
                for (int j=0;j<d;j++){
                    if (input[i]) continue;
                    float fast_adapt=dna[i].fast_adaptation_rate*(dna[i].dh_par_contrib*nerr[i*d+j]-dna[i].dh_chl_contrib*received_signal_chl[j]);
                    float energy_noise=gaussian_noise(0.0f,1.0f/(max(0.0f,e[i])+0.1f));
                    dh[i*d+j]=dt*(fast_adapt-dna[i].h_decay*h[i*d+j]+energy_noise);
                    nh[i*d+j]=tanh_mag*tanhf((nh[i*d+j]+dh[i*d+j])/tanh_mag);
                }
                nu[i]+=dt*((surprise/d)-u[i]); //uncertainty is a moving average of surprise
                //move a
                delta_rule(a, emitted_signal_par, nerr, dna[i].slow_adaptation_learning_rate_a, da, dna[i].a_decay, d, r, i);
                //move b (hypothesis here - align the errors, invert A=sort of "unify/harmonize" the whole network)
                delta_rule(b, emitted_signal_chl, err_diff, dna[i].slow_adaptation_learning_rate_b, db, dna[i].b_decay, d, r, i);
                //update energy
                if (input[i]){
                    //it's hard for input nodes to do well, and they provide the only signal source, so they should always be able to emit signal (scaled by e/(1+e) * 1/(1+u))
                    e[i]=100.0f;
                    nu[i]=0.0f;
                }
                e[i]+=dt*(1.0/(1.0f+u[i])*(dna[i].curiosity*dna[i].signal_income*sig_mag/(1.0f+sig_mag)-dna[i].stability*dna[i].surprise_tax*surprise)-dna[i].cost_of_thought*mag(dh, i*d, d)-dna[i].cost_of_complexity*(mag(a, i*d*r, d*r)+mag(b, i*d*r, d*r))-dna[i].e_decay*e[i]);
                e[i]=max(0.0f, e[i]);
                nu[i]=min(nu[i],10.0f);
            }
            for (int i=0;i<n;i++){
                //TODO: proper values
                E[i].step((e[i]/(e[i]+1.0f))*(1/(u[i]+1.0f)));
                V[i].step(1.0f/(e[i]+0.1f)+min(1.0f, mag(dh, i*d, d)));
                S[i].step(u[i]/(1.0f+u[i]));
                dna[i].dh_chl_contrib=1.0f*(1.0f-S[i].val);
                dna[i].dh_par_contrib=1.0f*(S[i].val);
                dna[i].fast_adaptation_rate=10.0f*E[i].val*V[i].val;
                dna[i].slow_adaptation_learning_rate_a=0.01f*E[i].val*V[i].val;
                dna[i].slow_adaptation_learning_rate_b=0.015f*E[i].val*V[i].val;
                dna[i].h_decay=0.05f*(1-S[i].val)*V[i].val;
                dna[i].e_decay=0.05f*V[i].val;
                dna[i].signal_income=1.0f*(1-S[i].val);
                dna[i].cost_of_complexity=0.2f*S[i].val*(1-E[i].val);
                dna[i].curiosity=1.0f*E[i].val;
                dna[i].stability=1.0f*(1-E[i].val);
                dna[i].surprise_tax=1.0f;//*(1-V[i].val)*(1-E[i].val)*S[i].val;
                dna[i].a_decay=0.05f*(1-S[i].val)*V[i].val;
                dna[i].b_decay=0.05f*(1-S[i].val)*V[i].val;
                dna[i].cost_of_thought=0.05f*(1-S[i].val)*E[i].val;
            }
            swap(h, nh);
            swap(err, nerr);
            swap(u, nu);
        }
        void update(int k){
            for (int i=0;i<k;i++) forward();
        }
        void step(){
            update(5);
        }
};
vector<bool> un_input(n, false); //input mask
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
}
vector<int> input_nodes{0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
float curp=0.2f;
int main(){
    for (auto xx:input_nodes){
        un_input[xx]=true;
    }
    cout<<"smallworld p: "<<curp<<endl;
    for (auto curtp:vector<string>{"lorenz", "sin", "brownian", "rossler", "fourier", "o-u"}){
        cout<<"RUNNING EXPERIMENT "<<curtp<<endl;
        for (int num=1;num<=2;num++){
            float curx=0.05f, cury=0.1f, curz=-0.025f;
            trait iE{0.9f, 0.5f, 0.4f, 0.9f, 0.0f, 0.05f, 0.05f, 0.4f};
            trait iV{0.9f, 0.65f, 0.6f, 0.85f, 0.0f, 0.05f, 0.1f, 0.6f};
            trait iS{0.1f, 0.3f, 0.2f, 0.95f, 0.0f, 0.05f, 0.03f, 0.2f};
            lupus sextus(genes(), iE, iV, iS, un_input);
            sextus.reset(curp);
            for (int i=0;i<100000;i++){
                //cout<<"STEP NUMBER: "<<i+1<<endl;
                update_data(curx, cury, curz, i, curtp);
                //cout<<"POS AT: "<<curx<<", "<<cury<<", "<<curz<<endl;
                for (auto xx:input_nodes){
                    sextus.h[xx*d+0]=curx/5.0f;
                    sextus.h[xx*d+1]=cury/5.0f;
                    sextus.h[xx*d+2]=curz/5.0f;
                }
                sextus.step();
                //cout<<"NODES EVS:"<<endl;
                int tot=0;
                for (int j=0;j<n;j++){
                    //cout<<"NODE "<<j+1<<": "<<endl;
                    //cout<<"E="<<setprecision(5)<<sextus.E[j].val<<"; V="<<setprecision(5)<<sextus.V[j].val<<"; S="<<setprecision(5)<<sextus.S[j].val<<"; energy="<<setprecision(5)<<sextus.e[j]<<"; uncertainty="<<setprecision(5)<<sextus.u[j]<<"; a_mag="<<setprecision(5)<<mag(sextus.a, j*d*r, d*r)<<"; b_mag="<<setprecision(5)<<mag(sextus.b, j*d*r, d*r)<<endl;
                    if (sextus.e[j]==0.0f) tot++;
                }
                if (tot==n-input_nodes.size()) {
                    cout<<"DEATH "<<num<<": "<<i<<endl;
                    break;
                }
            }
        }
    }
    return 0;
}
/*
clang++ -std=c++23 -O3 -Wall -DACCELERATE_NEW_LAPACK main_fast.cpp -framework Accelerate -o main_fast && ./main_fast
*/