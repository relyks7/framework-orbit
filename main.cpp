#include <bits/stdc++.h>
using namespace std;
int n; int d; int r; int deg; float eta; float alpha; float beta; float lambda; float dt; float omega; float gamma; float rho; float zeta; float theta; float kappa; float xi;
struct node{
    vector<float> h; //internal state
    vector<float> err; //error
    vector<float> out; //emission
    vector<float> z; //storage variable for oja/accumulated projected postsynaptic activity
    float e; //energy - leaky integrator
    float v; //voltage - liquid reservoir
    float stress; //aggregate of surprisal
    bool alive;
    vector<vector<float>> a;
    vector<vector<float>> b;
    vector<vector<float>> b_t;
};
struct edge{
    int from;
    float w;
};
vector<node> nodes;
vector<vector<edge>> adj;
vector<float> matvec(vector<vector<float>> a, vector<float> b){
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
vector<vector<float>> outer_prod(vector<float> a, vector<float> b){
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
void oja(vector<vector<float>> &a, vector<float> x, vector<float> z){
    vector<vector<float>> t1=outer_prod(x, z);
    vector<vector<float>> t2=outer_prod(matvec(a, z), z);
    for (int i=0;i<a.size();i++){
        for (int j=0;j<a[0].size();j++){
            a[i][j]+=eta*(t1[i][j]-t2[i][j]-gamma*a[i][j]);
        }
    }
}
void cleanup(){
    vector<int> new_index(n,-1);
    int cnt=0;
    for (int i=0;i<n;i++){
        if (!nodes[i].alive) continue;
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
void update(){
    vector<vector<float>> new_h(n, vector<float>(d,0));
    for (int i=0;i<n;i++){
        if (!nodes[i].alive) continue;
        new_h[i]=nodes[i].h;
        vector<float> z_in(r,0);
        for (auto x:adj[i]){
            if (!nodes[x.from].alive) continue;
            for (int j=0;j<r;j++){
                z_in[j]+=x.w*nodes[x.from].out[j];
            }
        }
        nodes[i].z=z_in;
        vector<float> sig=matvec(nodes[i].a, z_in);
        float surprise=0.0f;
        float h_mag=0.0f;
        for (int j=0;j<d;j++){
            nodes[i].err[j]=sig[j]-nodes[i].h[j];
            new_h[i][j]+=dt*(alpha*nodes[i].err[j]-lambda*nodes[i].h[j]);
            surprise+=nodes[i].err[j]*nodes[i].err[j];
            h_mag+=new_h[i][j]*new_h[i][j];
        }
        nodes[i].v+=dt*(surprise-rho*nodes[i].v);
        nodes[i].e+=dt*(h_mag-zeta*nodes[i].e);
        if (nodes[i].v>theta){
            nodes[i].out=matvec(nodes[i].b_t,nodes[i].err);
            nodes[i].v=0.0f;
            oja(nodes[i].a, new_h[i], nodes[i].z); //use new_h or current h?
        } else{
            fill(nodes[i].out.begin(), nodes[i].out.end(), 0.0f);
        }
        if (nodes[i].e<omega){
            nodes[i].alive=false;
        }
        nodes[i].stress+=1.0f-dt*kappa*nodes[i].stress;
    }
    for (int i=0;i<n;i++){
        if (!nodes[i].alive) continue;
        nodes[i].h=new_h[i];
    }
}
void mitosis(int p){ //no, kris did not mitose
    nodes[p].stress=0.0f;
    nodes[p].e/=2.0f;
    node child=nodes[p];
    adj.push_back({});
    for (int i=0;i<d;i++){
        for (int j=0;j<r;j++){
            child.a[i][j]+=gaussian_noise(0.0f, 0.01f); //TODO: make gaussian noise function
            float bgauss=gaussian_noise(0.0f, 0.01f);
            child.b[i][j]+=bgauss;
            child.b_t[j][i]+=bgauss;
        }
    }
    for (auto& x:adj[p]){
        x.w=x.w/2; //consider making mitoses imperfect
        adj[n].push_back({x.from, x.w});
    }
    for (int i=0;i<n;i++){
        for (auto& x:adj[i]){
            if (x.from==p){
                x.w/=2;
                adj[i].push_back({n, x.w});
            }
        }
    }
    nodes.push_back(child);
    adj.push_back({});
    node child=nodes[p];
    n++;
}
int main(){
    return 0;
}