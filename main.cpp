#include <bits/stdc++.h>
using namespace std;
mt19937 rng(random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
//in hindsight using greek letters wasn't the best idea
int n=4; //number of nodes
int d=4; //internal dimension
int r=64; //low-rank dimension
int deg=8; //maximum node degree
float dt=0.01; //delta time

//greek-letter parameters - global, if these can be mutated they directly lead to reward hacking (death, decay (of things that can cause death), cost (in energy) and drain (of energy based on population) should be constants)
float omega=0.01; //energy threshold for reaping/apoptosis
float sigma=50.0f; //stress threshold for apoptosis
float chi=0.01f; //energy cost of h (metabolic cost of thought)
float nu=2.0f; //energy cost of mitosing
float phi=1.61803398875; //golden ratio
float rho=0.05; //decay rate for voltage
float zeta=0.01; //decay rate for energy
float kappa=0.5; //decay rate for stress
float psi=500.0f; //parameter used in dynamic adjustment of energy drain

/*
a note on node types:
1. internal node - functions normally, this is the baseline.
2. motor node - no different to an internal node, except it's state is read to alter the environment.
3. input nodes - these nodes have their h force-set to a certain value. do not mitose or die. they also always fire.
4. prior nodes - these nodes have their h force-set to a prior and their sig force-set to the current state (e.g. h="i am full", sig="i am hungry") to generate error when a prior is not satisfied.
*/
struct node{
    vector<float> h; //internal state
    vector<float> err; //error
    vector<float> out; //emission
    vector<float> z; //storage variable for oja/accumulated projected postsynaptic activity
    float e; //energy - leaky integrator
    float v; //voltage - liquid reservoir
    float stress; //aggregate/spiking integrator of surprisal
    float u; //uncertainty/leaky integrator of/overall average surprisal
    bool alive;
    bool is_input;
    bool is_prior;
    vector<float> sig;
    vector<vector<float>> a;
    vector<vector<float>> b;
    vector<vector<float>> b_t;
    //greek-letter parameters - local, likely safe to mutate without enabling reward hacking
    float eta=1.0; //oja's learning rate **scaling parameter** (the learning rate is adapted based on stress)
    float alpha=0.3; //contribution of error/fast-timescale internal state adaptation rate
    float tau=15.0f; //stress threshold for mitosis
    float lambda=0.01f; //decay rate of h
    float upsilon=1.0f; //decay rate for compressed input z (leaky integrator)
    float theta=0.001; //voltage threshold for firing, triggering oja's, and spiking stress
    float omicron=0.0005f; //tax of magnitude of a in energy (complexity)
    float gamma=2.0f; //weight of magnitude of h in voltage
    float mu=2.0f; //how much error/surprise affects the fristonian side of energy "gain" (in this case loss)
    float delta=5.0f; //how much signal affects the "infomax" side of energy gain
    float xi=1.0f; //"curiosity" - how much of the node likes signal, and how much of it dislikes error (exploration/exploitation balance)
};
struct edge{
    int from;
    float w; //w is technically deprecated, but will keep here just in case it proves to be useful. also helps mark edges as dead.
};
vector<node> nodes{};
vector<deque<edge>> adj{};
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
void sanger(vector<vector<float>> &a, const vector<float>& x, const vector<float>& y, float lr){
    vector<float> t2(a[0].size(), 0);
    for (int i=0;i<a.size();i++){
        for (int j=0;j<a[0].size();j++){
            t2[j]+=a[i][j]*y[i];
            a[i][j]+=lr*y[i]*(x[j]-t2[j]);
        }
    }
}
void cleanup(){
    vector<int> new_index(n,-1);
    int cnt=0;
    for (int i=0;i<n;i++){
        if ((!nodes[i].alive)) continue;
        new_index[i]=cnt++;
    }
    vector<node> new_nodes{};
    for (int i=0;i<n;i++){
        if (!nodes[i].alive) continue;
        new_nodes.push_back(nodes[i]);
    }
    vector<deque<edge>> new_adj{};
    for (int i=0;i<n;i++){
        if (!nodes[i].alive) continue;
        new_adj.push_back({});
        for (auto x:adj[i]){
            if ((!nodes[x.from].alive) || x.w==0.0f) continue;
            new_adj[new_adj.size()-1].push_back({new_index[x.from], x.w});
        }
    }
    nodes=new_nodes;
    adj=new_adj;
    n=nodes.size();
}
void mitosis(int p){ //no, kris did not mitose
    float ps=1.0f/phi;
    float cs=1.0f/(phi*phi);
    nodes[p].stress=0.0f;
    node child=nodes[p];
    for (int i=0;i<d;i++){
        for (int j=0;j<r;j++){
            child.a[i][j]+=gaussian_noise(0.0f, 0.001f);
            float bgauss=gaussian_noise(0.0f, 0.001f);
            child.b[i][j]+=bgauss;
            child.b_t[j][i]+=bgauss;
        }
    }
    nodes.push_back(child);
    nodes[p].e=ps*nodes[p].e;
    nodes[n].e=cs*nodes[n].e;
    adj.push_back({});
    for (auto& x:adj[p]){
        adj[n].push_back({x.from, x.w});
    }
    for (int i=0;i<n;i++){
        int sz=adj[i].size();
        for (int j=0;j<sz;j++){
            if (adj[i][j].from==p){
                if (adj[i].size()==deg){
                    adj[i].pop_front();
                }
                adj[i].push_back({n, adj[i][j].w});
                // if (adj[i].size()>deg){
                //     auto weak=min_element(adj[i].begin(), adj[i].end(), [](edge a, edge b){return abs(a.w)<abs(b.w);});
                //     *weak=adj[i].back();
                //     adj[i].pop_back();
                // }
            }
        }
    }
    if (adj[p].size()==deg){
        adj[p].pop_front();
    }
    if (adj[n].size()==deg){
        adj[n].pop_front();
    }
    adj[p].push_back({n, 1.0f});
    adj[n].push_back({p, 1.0f});
    // if (adj[p].size()>deg){
    //     auto weak=min_element(adj[p].begin(), adj[p].end(), [](edge a, edge b){return abs(a.w)<abs(b.w);});
    //     *weak=adj[p].back();
    //     adj[p].pop_back();   
    // }
    // if (adj[n].size()>deg){
    //     auto weak=min_element(adj[n].begin(), adj[n].end(), [](edge a, edge b){return abs(a.w)<abs(b.w);});
    //     *weak=adj[n].back();
    //     adj[n].pop_back();
    // }
    nodes[n].eta+=gaussian_noise(0.0f, 0.001f);
    nodes[n].alpha+=gaussian_noise(0.0f, 0.001f);
    nodes[n].tau+=gaussian_noise(0.0f, 0.001f);
    nodes[n].lambda+=gaussian_noise(0.0f, 0.001f);
    nodes[n].upsilon+=gaussian_noise(0.0f, 0.001f);
    nodes[n].theta+=gaussian_noise(0.0f, 0.001f);
    nodes[n].omicron+=gaussian_noise(0.0f, 0.001f);
    nodes[n].gamma+=gaussian_noise(0.0f, 0.001f);
    nodes[n].mu+=gaussian_noise(0.0f, 0.001f);
    nodes[n].delta+=gaussian_noise(0.0f, 0.001f);
    nodes[n].xi+=gaussian_noise(0.0f, 0.001f);

    nodes[n].eta=min(max(nodes[n].eta, 0.0001f), 5.0f);
    nodes[n].alpha=min(max(nodes[n].alpha, 0.0001f), 5.0f);
    nodes[n].tau=min(max(nodes[n].tau, 1.0f), sigma-1.0f);
    nodes[n].lambda=min(max(nodes[n].lambda, 0.0001f), 10.0f);
    nodes[n].upsilon=min(max(nodes[n].upsilon, 0.0001f), 10.0f);
    nodes[n].theta=min(max(nodes[n].theta, 0.0001f), 1.0f);
    nodes[n].omicron=min(max(nodes[n].omicron, 0.0f), 0.1f);
    nodes[n].gamma=min(max(nodes[n].gamma, 0.0f), 10.0f);
    nodes[n].mu=min(max(nodes[n].mu, 0.001f), 20.0f);
    nodes[n].delta=min(max(nodes[n].delta, 0.001f), 20.0f);
    nodes[n].xi=min(max(nodes[n].xi, 0.0f), 1.0f);
    n++;
}
void update(){
    vector<vector<float>> new_h(n, vector<float>(d,0));
    for (int i=0;i<n;i++){
        if ((!nodes[i].alive)) continue;
        new_h[i]=nodes[i].h;
        vector<float> z_in(r,0);
        int iptn=0;
        if (!nodes[i].is_prior){
            for (auto x:adj[i]){
                if (!nodes[x.from].alive) continue;
                for (int j=0;j<r;j++){
                    z_in[j]+=x.w*nodes[x.from].out[j];
                }
                iptn++;
            }
            if (iptn>0){
                for (int j=0;j<r;j++){
                    z_in[j]/=iptn;
                }
            }
            for (int j=0;j<r;j++){
                nodes[i].z[j]+=dt*(z_in[j]-nodes[i].upsilon*nodes[i].z[j]);
            }
            nodes[i].sig=matvec(nodes[i].a, nodes[i].z);
        }
        float sig_mag=0.0f;
        for (int j=0;j<d;j++){
            nodes[i].sig[j]=tanhf(nodes[i].sig[j]);
            sig_mag+=nodes[i].sig[j]*nodes[i].sig[j];
        }
        float surprise=0.0f;
        float h_mag=0.0f;
        float a_mag=0.0f;
        for (int j=0;j<d;j++){
            nodes[i].err[j]=nodes[i].sig[j]-nodes[i].h[j];
            new_h[i][j]+=dt*(nodes[i].alpha*nodes[i].err[j]-nodes[i].lambda*nodes[i].h[j]);
            surprise+=nodes[i].err[j]*nodes[i].err[j];
            h_mag+=new_h[i][j]*new_h[i][j];
        }
        nodes[i].u+=dt*((surprise/d)-nodes[i].u);
        for (int j=0;j<d;j++){
            for (int k=0;k<r;k++){
                a_mag+=nodes[i].a[j][k]*nodes[i].a[j][k];
            }
        }
        nodes[i].v+=dt*((surprise+nodes[i].gamma*h_mag)/d-rho*nodes[i].v); //add h_mag, the brain can no longer just hold an internal representation and ignore the retinas/not broadcast it (dark room). if that representation is strong enough, it is forced to broadcast it, because merely holding that belief increases voltage.
        nodes[i].e+=dt*((1.0/(nodes[i].u+1.0f))*(nodes[i].xi*nodes[i].delta*sig_mag-(1.0f-nodes[i].xi)*nodes[i].mu*surprise)-nodes[i].omicron*a_mag-chi*h_mag-zeta*(1.0f+n/psi)*nodes[i].e); 
        nodes[i].stress-=dt*kappa*nodes[i].stress;
        if (nodes[i].is_input || nodes[i].is_prior){
            nodes[i].stress+=surprise*dt; //stress does not spike here otherwise the input/prior nodes undergo massive stress as they are locked on the input signal, causing issues - also, we want a continuous signal here
            nodes[i].out=matvec(nodes[i].b_t,nodes[i].err);
            if (!nodes[i].is_prior) sanger(nodes[i].a, nodes[i].z, new_h[i], nodes[i].eta*abs(tanhf(nodes[i].stress)));
        } else{
            if (nodes[i].v>nodes[i].theta){
                nodes[i].stress+=1.0f;
                nodes[i].out=matvec(nodes[i].b_t,nodes[i].h);
                nodes[i].v=0.0f;
                sanger(nodes[i].a, nodes[i].z, new_h[i], nodes[i].eta*abs(tanhf(nodes[i].stress))); //use new_h or current h?
            } else{
                fill(nodes[i].out.begin(), nodes[i].out.end(), 0.0f);
            }
            if (nodes[i].e<omega || nodes[i].stress>sigma){
                nodes[i].alive=false;
            }
        }
        for (int j=0;j<r;j++){
            nodes[i].out[j]=(1.0/(nodes[i].u+1.0f))*tanhf(nodes[i].out[j]); //out is gated by tanh and weighted by precision
        }
    }
    for (int i=0;i<n;i++){
        if ((!nodes[i].alive) || nodes[i].is_input || nodes[i].is_prior) continue;
        nodes[i].h=new_h[i];
    }
    int tn=n;
    for (int i=0;i<tn;i++){
        if ((!nodes[i].alive) || nodes[i].is_input || nodes[i].is_prior) continue;
        if (nodes[i].stress>nodes[i].tau && nodes[i].e>nu){ //if it is too stressed but also has enough energy, mitose
            nodes[i].e-=nu;
            mitosis(i);
        }
    }
}
void update_lorenz(float &x, float &y, float &z, float dt) {
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
int main(){
    //index0: input 1
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    0,0,0,0,
                    true,true,false,
                    vector<float>(d,0),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    //index1: input 2
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    0,0,0,0,
                    true,true,false,
                    vector<float>(d,0),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    //index2: input 3
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    0,0,0,0,
                    true,true,false,
                    vector<float>(d,0),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    //index3: internal 1
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    5.0f,0,0,0,
                    true,false,false,
                    vector<float>(d,0),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    //index4: motor 1
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    5.0f,0,0,0,
                    true,false,false,
                    vector<float>(d,0),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    //index5: prior 1 - boredom (overall, i should have some difficulty predictig things)
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    5.0f,0,0,0,
                    true,false,true,
                    vector<float>(d,0),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    for (int x=0;x<n;x++){
        for (int i=0;i<d;i++){
            for (int j=0;j<r;j++){
                nodes[x].a[i][j]=gaussian_noise(0.0f,0.1f);
                float bgauss=gaussian_noise(0.0f,0.125f);
                nodes[x].b[i][j]=bgauss;
                nodes[x].b_t[j][i]=bgauss;
            }
        }
    }
    for (int i=0;i<n;i++){
        adj.push_back({});
    }
    adj[0].push_back({3, 1.0f});
    adj[1].push_back({3, 1.0f});
    adj[2].push_back({3, 1.0f});
    adj[3].push_back({0, 1.0f});
    adj[3].push_back({1, 1.0f});
    adj[3].push_back({2, 1.0f});
    adj[3].push_back({4, 1.0f});
    adj[4].push_back({3, 1.0f});
    adj[5].push_back({3, 1.0f});
    int fin=10000;
    int count=0;
    float curx=0.05f, cury=0.1f, curz=-0.01f;
    for (float t=0;t<fin;t+=dt){
        count++;
        update_lorenz(curx, cury, curz, dt);
        if (count%100==0){

        }
        if (count % 1000 == 0) {
            cout << "t: " << fixed << setprecision(2) << t;
            for (int i=3;i<min(n,20);i++){
                cout<<" | N"<<i<<"_stress: "<<fixed<<setprecision(4)<<setw(5)<<nodes[i].stress;
                cout<<" | N"<<i<<"_energy: "<<fixed<<setprecision(4)<<setw(5)<<nodes[i].e;
                cout<<" | N"<<i<<"_voltage: "<<fixed<<setprecision(4)<<setw(5)<<nodes[i].v;
            }
            if (n>20){
                cout<<"... ("<<n<<" total nodes)";
            } else{
                cout<<" ("<<n<<" total nodes)";
            }
            cout<<endl;
        }
        if (count%100==0){
            cleanup();
        }
        if (count%100==0){
            if (((count^(count<<5)^(count<<7)^(count<<11)<<(count<<17)^(count/7))%100)<5){
                int r1=((count^(count<<5)^(count<<7)^(count<<11)^(count<<17)^(count/7))*0xbf58476d1ce4e5b9)%n;
                int r2=((count^(count<<6)^(count<<2)^(count>>5)^(count<<12)^(count/3))*0x94d049bb133111eb)%n;
                if (r1==r2) continue;
                adj[r1].push_back({r2, 1.0f});
            }
        }
        // fill(nodes[0].h.begin(), nodes[0].h.end(), sin(t)); 
        // fill(nodes[1].h.begin(), nodes[1].h.end(), cos(t));
        // fill(nodes[2].h.begin(), nodes[2].h.end(), 0.0f);

        // fill(nodes[0].h.begin(), nodes[0].h.end(), (sin(t) + 2.0f * sin(2.0f * t)) / 3.0f); 
        // fill(nodes[1].h.begin(), nodes[1].h.end(), (cos(t) - 2.0f * cos(2.0f * t)) / 3.0f);
        // fill(nodes[2].h.begin(), nodes[2].h.end(), -sin(3.0f * t));

        // fill(nodes[0].h.begin(), nodes[0].h.end(),curx/20.0f); 
        // fill(nodes[1].h.begin(), nodes[1].h.end(),cury/20.0f);
        // fill(nodes[2].h.begin(), nodes[2].h.end(),(curz-25.0f)/25.0f);
        
        // fill(nodes[0].h.begin(), nodes[0].h.end(),gaussian_noise(0.0f, 10.0f)); 
        // fill(nodes[1].h.begin(), nodes[1].h.end(),gaussian_noise(0.0f, 10.0f));
        // fill(nodes[2].h.begin(), nodes[2].h.end(),gaussian_noise(0.0f, 10.0f));
        
        fill(nodes[5].h.begin(), nodes[5].h.end(), 0.5);
        float avgu=0.0f;
        int talive=0;
        for (int i=0;i<n;i++){
            if (!nodes[i].alive) continue;
            talive++;
            avgu+=nodes[i].u;
        }
        avgu/=talive;
        fill(nodes[5].sig.begin(), nodes[5].sig.end(), avgu);
        update();
    }
    return 0;
}