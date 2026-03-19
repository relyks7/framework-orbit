#include <bits/stdc++.h>
using namespace std;
int n; int d; int r; float eta; float beta; float lambda; float dt; float omega; float gamma; vector<vector<float>> b; vector<vector<float>> b_t;
struct node{
    vector<float> h;
    float e;
    bool alive;
    vector<vector<float>> a;
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
void oja(vector<vector<float>> &a, vector<float> x, vector<float> y){
    vector<float> z=matvec(b_t, y);
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
            if (!nodes[x.from].alive) continue;
            new_adj[new_adj.size()-1].push_back({new_index[x.from], x.w});
        }
    }
    nodes=new_nodes;
    adj=new_adj;
    n=nodes.size();
}
void udpate_h(){
    vector<vector<float>> new_h(n, vector<float>(d,0));
    for (int i=0;i<n;i++){
        if (!nodes[i].alive) continue;
        new_h[i]=nodes[i].h;
        for (int j=0;j<d;j++){
            new_h[i][j]-=dt*lambda*nodes[i].h[j];
        }
        for (auto x:adj[i]){
            if (!nodes[x.from].alive) continue;
            for (int j=0;j<d;j++){
                new_h[i][j]+=dt*x.w*nodes[x.from].h[j];
            }
        }
    }
    for (int i=0;i<n;i++){
        nodes[i].h=new_h[i];
    }
}
void udpate_e(){
    for (int i=0;i<n;i++){
        if (!nodes[i].alive) continue;
        float act=0.0f;
        for (int j=0;j<d;j++){
            act+=nodes[i].h[j]*nodes[i].h[j];
        }
        act=sqrt(act);
        nodes[i].e=beta*nodes[i].e+(1-beta)*act;
        if (nodes[i].e<omega){
            nodes[i].alive=false;
        }
    }
}
int main(){
    n=1024;
    d=128;
    nodes.assign(n, {vector<float>(d,0), 0.0f, true});
    adj.assign(n, vector<edge>{});
}