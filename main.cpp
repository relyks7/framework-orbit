#include <bits/stdc++.h>
using namespace std;
mt19937 rng(random_device{}());
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
//in hindsight using greek letters wasn't the best idea
int n=4; //number of nodes
int d=64; //internal dimension
int r=4; //low-rank dimension
int deg=8; //maximum node degree
float dt=0.01; //delta time
float eta=0.1; //oja's learning rate **scaling parameter** (the learning rate is adapted based on stress)
float alpha=0.3; //contribution of error/fast-timescale internal state adaptation rate
float tau=15.0f; //stress threshold for mitosis
float lambda=0.01f; //decay rate of h
float omega=0.01; //energy threshold for reaping/apoptosis
float xi=0.001; //decay rate for a
float rho=0.15; //decay rate for voltage
float zeta=0.075; //decay rate for energy
float theta=0.05; //voltage threshold for firing, triggering oja's, and spiking stress
float kappa=0.05; //decay rate for stress
float upsilon=1.0f; //decay rate for compressed input z (leaky integrator)
float omicron=0.01f; //weight of magnitude of a in energy
float gamma=2.0f; //weight of magnitude of h in voltage
float chi=0.05f; //energy cost of h (metabolic cost of thought)
float psi=500.0f; //parameter used in dynamic adjustment of energy drain
float mu=0.3f; //when a node dies, how much of it's learned a goes to neighbours (porportional to # of live neighbours)?
float nu=0.1f; //energy cost of mitosing
float sigma=2*tau; //stress threshold for apoptosis
float phi=1.61803398875; //golden ratio
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
    vector<vector<float>> a;
    vector<vector<float>> b;
    vector<vector<float>> b_t;
};
struct edge{
    int from;
    float w;
};
vector<node> nodes{};
vector<vector<edge>> adj{};
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
    vector<float> t2(a.size(), 0);
    for (int i=0;i<a.size();i++){
        for (int j=0;j<a[0].size();j++){
            t2[i]+=a[i][j]*y[j];
            a[i][j]+=eta*lr*y[j]*(x[i]-t2[i]);
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
    vector<vector<edge>> new_adj{};
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
            child.a[i][j]+=gaussian_noise(0.0f, 0.1f);
            float bgauss=gaussian_noise(0.0f, 0.1f);
            child.b[i][j]+=bgauss;
            child.b_t[j][i]+=bgauss;
        }
    }
    nodes.push_back(child);
    nodes[p].e=ps*nodes[p].e;
    nodes[n].e=cs*nodes[n].e;
    adj.push_back({});
    for (auto& x:adj[p]){
        // x.w=x.w/2; //consider making mitoses imperfect to introduce a bit more asymmetry
        //actually don't halve here, it lets the network mitose until it cuts the error by just cutting the input
        if (gaussian_noise(0.0f, 1.0f)>0){
            adj[n].push_back({x.from, x.w});
            x.w=0.0f;
        }
    }
    for (int i=0;i<n;i++){
        int sz=adj[i].size();
        for (int j=0;j<sz;j++){
            if (adj[i][j].from==p){
                if (gaussian_noise(0.0f, 1.0f)>0){
                    adj[i].push_back({n, adj[i][j].w});
                    adj[i][j].w=0;
                    if (adj[i].size()>deg){
                        auto weak=min_element(adj[i].begin(), adj[i].end(), [](edge a, edge b){return abs(a.w)<abs(b.w);});
                        *weak=adj[i].back();
                        adj[i].pop_back();
                    }
                }
            }
        }
    }
    adj[p].push_back({n, -0.7f});
    adj[n].push_back({p, -0.7f});
    if (adj[p].size()>deg){
        auto weak=min_element(adj[p].begin(), adj[p].end(), [](edge a, edge b){return abs(a.w)<abs(b.w);});
        *weak=adj[p].back();
        adj[p].pop_back();   
    }
    if (adj[n].size()>deg){
        auto weak=min_element(adj[n].begin(), adj[n].end(), [](edge a, edge b){return abs(a.w)<abs(b.w);});
        *weak=adj[n].back();
        adj[n].pop_back();
    }
    n++;
}
void update(){
    vector<vector<float>> new_h(n, vector<float>(d,0));
    for (int i=0;i<n;i++){
        if ((!nodes[i].alive)) continue;
        new_h[i]=nodes[i].h;
        vector<float> z_in(r,0);
        for (auto x:adj[i]){
            if (!nodes[x.from].alive) continue;
            for (int j=0;j<r;j++){
                z_in[j]+=x.w*nodes[x.from].out[j];
            }
        }
        for (int j=0;j<r;j++){
            nodes[i].z[j]+=dt*(z_in[j]-upsilon*nodes[i].z[j]);
        }
        vector<float> sig=matvec(nodes[i].a, nodes[i].z);
        for (int j=0;j<d;j++){
            sig[j]=tanhf(sig[j]);
        }
        float surprise=0.0f;
        float h_mag=0.0f;
        float a_mag=0.0f;
        for (int j=0;j<d;j++){
            nodes[i].err[j]=sig[j]-nodes[i].h[j];
            new_h[i][j]+=dt*(alpha*nodes[i].err[j]-lambda*nodes[i].h[j]);
            surprise+=nodes[i].err[j]*nodes[i].err[j];
            h_mag+=new_h[i][j]*new_h[i][j];
        }
        nodes[i].u+=dt*((surprise/d)-nodes[i].u);
        for (int j=0;j<d;j++){
            for (int k=0;k<r;k++){
                a_mag+=nodes[i].a[j][k]*nodes[i].a[j][k];
            }
        }
        nodes[i].v+=dt*((surprise+gamma*h_mag)/d-rho*nodes[i].v); //add h_mag, the brain can no longer just hold an internal representation and ignore the retinas (dark room). if that representation is strong enough, it is forced to broadcast it, because merely holding that belief increases voltage.
        nodes[i].e+=dt*((1.0/(nodes[i].u+1.0f))*omicron*a_mag-chi*h_mag-zeta*(1.0f+n/psi)*nodes[i].e); //must also weigh in a_mag so that achieving perfection doesn't kill it if it's useful. also, thought does not drive energy - thought should burn energy. a increases energy - having structure means that it should last. also more people = faster energy decay
        nodes[i].stress-=dt*kappa*nodes[i].stress;
        if (nodes[i].is_input){
            nodes[i].stress+=surprise*dt; //stress does not spike here otherwise the sensory input nodes undergo massive stress as they are locked on the input signal, causing issues
            nodes[i].out=matvec(nodes[i].b_t,nodes[i].err);
            sanger(nodes[i].a, new_h[i], nodes[i].z, abs(tanhf(nodes[i].stress)));
        } else{
            if (nodes[i].v>theta){
                nodes[i].stress+=1.0f;
                nodes[i].out=matvec(nodes[i].b_t,nodes[i].h);
                nodes[i].v=0.0f;
                sanger(nodes[i].a, new_h[i], nodes[i].z, abs(tanhf(nodes[i].stress))); //use new_h or current h?
            } else{
                fill(nodes[i].out.begin(), nodes[i].out.end(), 0.0f);
            }
            if (nodes[i].e<omega || nodes[i].stress>sigma){
                int ln=0;
                for (auto x:adj[i]){
                    if (nodes[x.from].alive){
                        ln++;
                    }
                }
                for (auto x:adj[i]){
                    if (nodes[x.from].alive){
                        for (int j=0;j<d;j++){
                            for (int k=0;k<r;k++){
                                nodes[x.from].a[j][k]+=(nodes[i].a[j][k]*mu)/ln; //let the live neighbours carry some of the representation of the dead
                            }
                        }
                    }
                }
                nodes[i].alive=false;
            }
        }
        for (int j=0;j<d;j++){
            for (int k=0;k<r;k++){
                nodes[i].a[j][k]-=dt*xi*nodes[i].a[j][k];
            }
        }
        for (int j=0;j<r;j++){
            nodes[i].out[j]=(1.0/(nodes[i].u+1.0f))*tanhf(nodes[i].out[j]); //out is gated by tanh and weighted by precision
        }
    }
    for (int i=0;i<n;i++){
        if ((!nodes[i].alive) || nodes[i].is_input) continue;
        nodes[i].h=new_h[i];
    }
    int tn=n;
    for (int i=0;i<tn;i++){
        if ((!nodes[i].alive) || nodes[i].is_input) continue;
        if (nodes[i].stress>tau && nodes[i].e>nu){ //if it is too stressed but also has enough energy, mitose
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
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    0,0,0,0,
                    true,true,
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    0,0,0,0,
                    true,true,
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    0,0,0,0,
                    true,true,
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    nodes.push_back({vector<float>(d,0),
                    vector<float>(d,0),
                    vector<float>(r,0),
                    vector<float>(r,0),
                    5.0f,0,0,0,
                    true,false,
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(d,vector<float>(r,0)),
                    vector<vector<float>>(r,vector<float>(d,0))});
    for (int x=0;x<n;x++){
        for (int i=0;i<d;i++){
            for (int j=0;j<r;j++){
                nodes[x].a[i][j]=gaussian_noise(0.0f,0.5f);
                float bgauss=gaussian_noise(0.0f,0.125f);
                nodes[x].b[i][j]=bgauss;
                nodes[x].b_t[j][i]=bgauss;
            }
        }
    }
    for (int i=0;i<n;i++){
        adj.push_back({});
    }
    adj[0].push_back({3, 0.9f});
    adj[1].push_back({3, 0.9f});
    adj[2].push_back({3, 0.9f});
    adj[3].push_back({0, 0.9f});
    adj[3].push_back({1, 0.9f});
    adj[3].push_back({2, 0.9f});
    int fin=10000;
    int count=0;
    float curx=0.05f, cury=0.1f, curz=-0.01f;
    for (float t=0;t<fin;t+=dt){
        count++;
        update_lorenz(curx, cury, curz, dt);
        if (count % 1000 == 0) {
            cout << "t: " << fixed << setprecision(2) << t;
            for (int i=3;i<min(n,20);i++){
                cout<<" | N"<<i<<"_stress: "<<fixed<<setprecision(4)<<setw(5)<<nodes[i].stress;
                cout<<" | N"<<i<<"_energy: "<<fixed<<setprecision(4)<<setw(5)<<nodes[i].e;
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
        fill(nodes[0].h.begin(), nodes[0].h.end(), sin(t)); 
        fill(nodes[1].h.begin(), nodes[1].h.end(), cos(t));
        fill(nodes[2].h.begin(), nodes[2].h.end(), 0.0f);

        // fill(nodes[0].h.begin(), nodes[0].h.end(), (sin(t) + 2.0f * sin(2.0f * t)) / 3.0f); 
        // fill(nodes[1].h.begin(), nodes[1].h.end(), (cos(t) - 2.0f * cos(2.0f * t)) / 3.0f);
        // fill(nodes[2].h.begin(), nodes[2].h.end(), -sin(3.0f * t));

        // fill(nodes[0].h.begin(), nodes[0].h.end(),curx/20.0f); 
        // fill(nodes[1].h.begin(), nodes[1].h.end(),cury/20.0f);
        // fill(nodes[2].h.begin(), nodes[2].h.end(),(curz-25.0f)/25.0f);
        update();
    }
    return 0;
}