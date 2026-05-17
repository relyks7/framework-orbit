#include <bits/stdc++.h>
#include <Accelerate/Accelerate.h>
using namespace std;
#define all(v) v.begin(), v.end()
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
float delta_clamp=1.0f;
float fr=0.1f; //food radius
float speed=2.0f;
float bound=500.0f;
float within_dist=0.2f; //tolerance for differentiating direcion
float phi=1.618033988749895;
void matvec(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int idx1, int idx2){
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*m), 1, 0.0f, c.data(), 1);
}
void matvec_transpose(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int idx1, int idx2){
    cblas_sgemv(CblasRowMajor, CblasTrans, n, m, 1.0f, a.data()+(idx1*n*m), m, b.data()+(idx2*m), 1, 0.0f, c.data(), 1);
}
vector<float> outer_prod(const vector<float>& a, const vector<float>& b){
    int n=a.size();
    int m=b.size();
    vector<float> c(n*m);
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            c[i*m+j]=a[i]*b[j];
        }
    }
    return c;
}
float mag(const vector<float>&a, int start_idx, int size){
    return cblas_sdot(size, a.data()+start_idx, 1, a.data()+start_idx, 1)/size;
}
void delta_rule(vector<float> &a, const vector<float>&x, const vector<float>&err, float lr, vector<float>& da, float a_dec, int n, int m, int idx){
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            da[idx*m*n+i*m+j]=max(-delta_clamp, min(delta_clamp, -lr*dt*err[idx*n+i]*x[j]));
            a[idx*m*n+i*m+j]+=-dt*a_dec*a[idx*m*n+i*m+j]-da[idx*m*n+i*m+j];
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
        vector<float> nu, nh, nerr, na, nb;
        vector<float> emitted_signal_chl;
        vector<float> chl_signal;
        vector<float> received_signal_chl;
        genes dna;
        float curx;
        float cury;
        float foodx;
        float foody;
        float hunger;
        int curdir;
        bool alive;
        int lifetime;
        float avg_life;
        lupus(const genes& rdna, const vector<bool>& uinput){
            adj.assign(n,vector<int>{}); radj.assign(n,vector<int>{}); dna=rdna; input=uinput;
            h.assign(n*d, 0);
            dh.assign(n*d, 0);
            e.assign(n, starting_energy);
            err.assign(n*d, 0);
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
        }
        void reset(float p){
            adj.assign(n,vector<int>{}); radj.assign(n,vector<int>{});
            generate_smallworld(p, adj, radj);
            alive=true;
            lifetime=0;
            for (int i=0;i<n;i++){
                for (int j=0;j<d;j++){
                    for (int k=0;k<r;k++){
                        a[i*d*r+j*r+k]=gaussian_noise(0.0f, 1.0f/sqrtf(r));
                        b[i*d*r+j*r+k]=gaussian_noise(0.0f, 1.0f/sqrtf(r));
                    }
                }
            }
            curx=bound/4;
            cury=bound/4;
            foodx=bound/2;
            foody=bound/2;
            hunger=1.0f;
        }
        void update(){
            nu=u;
            nh=h;
            nerr=err;
            na=a;
            nb=b;
            for (int i=0;i<n;i++){
                //step 1: aggregate belief from parents
                fill(all(emitted_signal_par), 0.0f);
                for (auto par:radj[i]){
                    matvec_transpose(b, h, par_signal, d, r, par, par);
                    for (int j=0;j<r;j++){
                        emitted_signal_par[j]+=1.0f/(1.0f+u[par])*par_signal[j];
                    }
                }
                matvec(a, emitted_signal_par, received_signal_par ,d, r, i, 0);
                float surprise=0.0f; //surprise=squared sum of error
                for (int j=0;j<d;j++){
                    if (radj[i].size()>0) received_signal_par[j]/=radj[i].size();
                    nerr[i*d+j]=received_signal_par[j]-h[i*d+j];
                    //XXX: temporary addon to add hunger prior
                    if (i==0 && j==0){
                        nerr[i*d+j]=((1.0f-hunger)*sqrtf(pow(curx-foodx, 2)+pow(cury-foody, 2)))*phi;
                    }
                    surprise+=nerr[i*d+j]*nerr[i*d+j];
                }
                nu[i]+=dt*((surprise/d)-u[i]); //uncertainty is a moving average of surprise
                //step 2: aggregate error from children
                fill(all(emitted_signal_chl), 0.0f);
                for (auto chl:adj[i]){
                    matvec_transpose(a, err, chl_signal, d, r, chl, chl);
                    for (int j=0;j<r;j++){
                        emitted_signal_chl[j]+=1.0f/(1.0f+u[chl])*chl_signal[j];
                    }
                }
                matvec(b, emitted_signal_chl, received_signal_chl, d, r, i, 0);
                for (int j=0;j<d;j++){
                    if (adj[i].size()>0) received_signal_chl[j]/=adj[i].size();
                }
                float sig_mag=mag(received_signal_par, 0, received_signal_par.size())+mag(received_signal_chl, 0, received_signal_chl.size()); //magnitude of received signal
                //move h
                for (int j=0;j<d;j++){
                    if (input[i]) continue;
                    float fast_adapt=dna.fast_adaptation_rate*(dna.dh_par_contrib*nerr[i*d+j]-dna.dh_chl_contrib*received_signal_chl[j]);
                    float energy_noise=gaussian_noise(0.0f,1.0f/(max(0.0f,e[i])+0.1f));
                    dh[i*d+j]=dt*(fast_adapt-dna.h_decay*h[i*d+j]+energy_noise);
                    nh[i*d+j]+=dh[i*d+j];
                }
                //move a
                delta_rule(na, emitted_signal_par, nerr, dna.slow_adaptation_learning_rate_a/(e[i]+0.5), da, dna.a_decay, d, r, i);
                //move b (hypothesis here - align the errors, invert A=sort of "unify/harmonize" the whole network)
                delta_rule(nb, emitted_signal_chl, nerr, dna.slow_adaptation_learning_rate_b/(e[i]+0.5), db, dna.b_decay, d, r, i);
                //update energy
                e[i]+=dt*(1.0/(1.0f+u[i])*(dna.curiosity*dna.signal_income*sig_mag-dna.stability*dna.surprise_tax*surprise)-dna.cost_of_thought*mag(dh, i*d, d)-dna.cost_of_complexity*mag(da, i*d*r, d*r)-dna.e_decay*e[i]);
                e[i]=max(0.0f, e[i]);
                nu[i]=min(nu[i],10.0f);
            }
            swap(h, nh);
            swap(err, nerr);
            swap(a, na);
            swap(b, nb);
            swap(u, nu);
        }
        void step(){
            lifetime++;
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
                    dh[1*d+i]=1.0f-h[1*d+i];
                    h[1*d+i]=1.0f;
                } else{
                    dh[1*d+i]=0.0f-h[1*d+i];
                    h[1*d+i]=0.0f;
                }
            }
            update();
            curx+=dt*speed*tanhf(h[2*d+0]);
            cury+=dt*speed*tanhf(h[2*d+1]);
            curx=max(0.0f,min(bound,curx));
            cury=max(0.0f,min(bound,cury));
            if (sqrtf(pow(curx-foodx, 2)+pow(cury-foody, 2))<fr){
                hunger+=0.3;
                foodx=disf(rng)*bound;
                foody=disf(rng)*bound;
            }
            hunger-=0.05*dt;
            hunger=max(0.0f,min(1.0f,hunger));
            if (hunger==0.0f) alive=false;
        }
        void run_life(int lives, float p){
            vector<float> lifetimes{};
            for (int i=0;i<lives;i++){
                cout<<"LIFE "<<i+1<<"\n";
                reset(p);
                while (alive){
                    step();
                }
                lifetimes.push_back(lifetime);
            }
            avg_life=accumulate(all(lifetimes), 0.0f)/lives;
            cout<<"AVERAGE LIFESPAN: "<<avg_life<<"\n";
        }
};
vector<bool> un_input(n, false);
genes crossover(genes dna1, genes dna2){
    return genes{
        (disf(rng)>0.5f)?dna1.dh_chl_contrib:dna2.dh_chl_contrib,
        (disf(rng)>0.5f)?dna1.dh_par_contrib:dna2.dh_par_contrib,
        (disf(rng)>0.5f)?dna1.fast_adaptation_rate:dna2.fast_adaptation_rate,
        (disf(rng)>0.5f)?dna1.slow_adaptation_learning_rate_a:dna2.slow_adaptation_learning_rate_a,
        (disf(rng)>0.5f)?dna1.slow_adaptation_learning_rate_b:dna2.slow_adaptation_learning_rate_b,
        (disf(rng)>0.5f)?dna1.h_decay:dna2.h_decay,
        (disf(rng)>0.5f)?dna1.e_decay:dna2.e_decay,
        (disf(rng)>0.5f)?dna1.signal_income:dna2.signal_income,
        (disf(rng)>0.5f)?dna1.cost_of_thought:dna2.cost_of_thought,
        (disf(rng)>0.5f)?dna1.cost_of_complexity:dna2.cost_of_complexity,
        (disf(rng)>0.5f)?dna1.curiosity:dna2.curiosity,
        (disf(rng)>0.5f)?dna1.stability:dna2.stability,
        (disf(rng)>0.5f)?dna1.surprise_tax:dna2.surprise_tax,
        (disf(rng)>0.5f)?dna1.a_decay:dna2.a_decay,
        (disf(rng)>0.5f)?dna1.b_decay:dna2.b_decay
    };
}
genes mutate(genes dna1, float p, float mutation_rate){
    dna1.dh_chl_contrib*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.dh_par_contrib*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.fast_adaptation_rate*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.slow_adaptation_learning_rate_a*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.slow_adaptation_learning_rate_b*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.h_decay*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.e_decay*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.signal_income*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.cost_of_thought*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.cost_of_complexity*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.curiosity*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.stability*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.surprise_tax*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.a_decay*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    dna1.b_decay*=(disf(rng)<p)?exp(gaussian_noise(0.0f, mutation_rate)):1.0f;
    return dna1;
}
genes random_gene(float magnitude){
    return genes{
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng),
        disf(rng)
    };
}
int population=100;
int survivors=5;
int cross_cnt=75;
int mutation_cnt=population-survivors-cross_cnt;
int epochs=10;
uniform_int_distribution<int> disi1(0,survivors-1);
uniform_int_distribution<int> disi2(0,survivors-2);
int main(){
    un_input[1]=true;
    vector<lupus> pack{};
    vector<lupus> new_pack{};
    for (int i=0;i<population;i++){
        pack.push_back(lupus(random_gene(1.0f), un_input));
    }
    for (int i=0;i<epochs;i++){
        cout<<"EPOCH "<<i+1<<"\n";
        new_pack.clear();
        for (int i=0;i<population;i++) {cout<<"WOLF #"<<i+1<<"\n"; pack[i].run_life(5, 0.1f);}
        sort(all(pack), [](lupus& l1, lupus& l2){return l1.avg_life>l2.avg_life;});
        for (int j=0;j<survivors;j++) new_pack.push_back(pack[j]);
        for (int j=0;j<cross_cnt;j++){
            int idx1=disi1(rng);
            int idx2=disi2(rng);
            if (idx2>=idx1) idx2++;
            lupus cand1=new_pack[idx1];
            lupus cand2=new_pack[idx2];
            new_pack.push_back(lupus(crossover(cand1.dna, cand2.dna), un_input));
        }
        for (int j=0;j<mutation_cnt;j++){
            new_pack.push_back(lupus(mutate(new_pack[disi1(rng)].dna, 0.2f, 1.0f), un_input));
        }
        pack=new_pack;
    }
    return 0;
}