
//shortest path chooser
static constexpr ll INF = numeric_limits<long long>::max();
static constexpr ll NINF = numeric_limits<long long>::min();
static constexpr int INVALID = -1;
template<typename T, T INFV = numeric_limits<T>::max()/2, int INV = INVALID>
struct shortest_path {
    int V, E;
    bool single_positive_weight;
    T wmin, wmax;

    vpii tos;
    vii head;
    vector<tuple<int,int,T>> edges;

    vll dist;
    vii prev;

    shortest_path(int _V = 0)
        : V(_V), E(0), single_positive_weight(true), wmin(0), wmax(0) {}

    void add_edge(int u, int v, T w) {
        assert(u >= 0 && u < V && v >= 0 && v < V);
        edges.pb({u, v, w});
        ++E;
        if (w > 0 && wmax > 0 && wmax != w) single_positive_weight = false;
        chmin(wmin, w);
        chmax(wmax, w);
    }
    void add_bi_edge(int u, int v, T w) { add_edge(u,v,w); add_edge(v,u,w); }

    void build_() {
        if (len(tos)==E && len(head)==V+1) return;
        tos.assign(E, {});
        head.assign(V+1,0);
        each(e, edges) ++head[get<0>(e)+1];
        rep(i,V) head[i+1] += head[i];
        auto cur = head;
        each(e, edges) {
            int u,v; T w; tie(u,v,w)=e;
            tos[cur[u]++] = {v,(int)w};
        }
    }

    template<class Heap = priority_queue<pair<T,int>, vector<pair<T,int>>, greater<>>> 
    void dijkstra(int s, int t = INV) {
        build_();
        dist.assign(V, INFV); prev.assign(V, INV);
        dist[s]=0;
        Heap pq; pq.emplace(0,s);
        while(!pq.empty()){
            auto [d,u]=pq.top(); pq.pop();
            if(u==t) return;
            if(dist[u]<d) continue;
            rep(e,head[u],head[u+1]){
                auto [v,w]=tos[e]; T nd=d+w;
                if(dist[v]>nd){ dist[v]=nd; prev[v]=u; pq.emplace(nd,v); }
            }
        }
    }

    void solve(int s, int t = INV) {
        if(wmin>=0) {
            if(single_positive_weight) zero_one_bfs(s,t);
            else if(wmax<=10) dial(s,t);
            else if((ll)V*V < (E<<4)) dijkstra_vquad(s,t);
            else dijkstra(s,t);
        } else bellman_ford(s,V);
    }

    vector<int> retrieve_path(int g) const {
        if(dist[g]==INFV) return {};
        vector<int> p;
        for(int u=g; u!=INV; u=prev[u]) p.pb(u);
        reverse(all(p));
        return p;
    }

    void dijkstra_vquad(int s, int t = INV) {
        build_();
        dist.assign(V, INFV); prev.assign(V, INV);
        dist[s]=0;
        vbl used(V,false);
        while (true) {
            int u = INV; T best = INFV;
            rep(i,V) if(!used[i] && dist[i]<best){ u=i; best=dist[i]; }
            if(u==INV || u==t) break;
            used[u]=true;
            rep(e,head[u],head[u+1]){
                auto [v,w]=tos[e];
                if(dist[v]>dist[u]+w){ dist[v]=dist[u]+w; prev[v]=u; }
            }
        }
    }

    bool bellman_ford(int s, int nb) {
        build_();
        dist.assign(V, INFV); prev.assign(V, INV);
        dist[s]=0;
        rep(loop,nb) {
            bool upd=false;
            rep(u,V) if(dist[u]!=INFV) rep(e,head[u],head[u+1]){
                auto [v,w]=tos[e]; T nd=dist[u]+w;
                if(dist[v]>nd){ dist[v]=nd; prev[v]=u; upd=true; }
            }
            if(!upd) return true;
        }
        return false;
    }

    void spfa(int s) {
        build_();
        dist.assign(V, INFV); prev.assign(V, INV);
        dist[s]=0;
        deque<int> q; vbl inq(V,false);
        q.pb(s); inq[s]=true;
        while(!q.empty()){
            int u=q.front(); q.pop_front(); inq[u]=false;
            rep(e,head[u],head[u+1]){
                auto [v,w]=tos[e]; T nd=dist[u]+w;
                if(dist[v]>nd){ dist[v]=nd; prev[v]=u;
                    if(!inq[v]){
                        if(!q.empty() && nd<dist[q.front()]) q.push_front(v);
                        else q.pb(v);
                        inq[v]=true;
                    }
                }
            }
        }
    }

    void zero_one_bfs(int s, int t = INV) {
        build_();
        dist.assign(V, INFV); prev.assign(V, INV);
        dist[s]=0;
        vector<int> q(4*V);
        int l=2*V, r=2*V;
        q[r++]=s;
        while(l<r){
            int u=q[l++]; if(u==t) return;
            rep(e,head[u],head[u+1]){
                auto [v,w]=tos[e]; T nd=dist[u]+w;
                if(dist[v]>nd){ dist[v]=nd; prev[v]=u;
                    if(w) q[r++]=v; else q[--l]=v;
                }
            }
        }
    }

    void dial(int s, int t = INV) {
        build_();
        dist.assign(V, INFV); prev.assign(V, INV);
        dist[s]=0;
        vector<vector<pair<int,T>>> buck(wmax+1);
        buck[0].eb(s,0);
        int inq_cnt=1;
        for(int cur=0; inq_cnt; ++cur){
            if(cur>=(int)buck.size()) cur=0;
            while(!buck[cur].empty()){
                auto [u,du]=buck[cur].back(); buck[cur].pop_back(); --inq_cnt;
                if(u==t) return;
                if(dist[u]<du) continue;
                rep(e,head[u],head[u+1]){
                    auto [v,w]=tos[e]; T nd=du+w;
                    if(dist[v]>nd){ dist[v]=nd; prev[v]=u;
                        int idx=(cur+w)%(wmax+1);
                        buck[idx].eb(v,nd); ++inq_cnt;
                    }
                }
            }
        }
    }
};
