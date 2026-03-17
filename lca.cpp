

// LCA — Binary Lifting
struct LCA {
    ll n, LOG;
    vvll adj;
    vvll par;     // par[v][j] = 2^j-th ancestor
    vll depth;

    LCA(ll n) : n(n) {
        LOG = __lg(n) + 1;
        adj.assign(n, {});
        par.assign(n, vll(LOG, -1));
        depth.assign(n, 0);
    }

    // add undirected edge
    void add_edge(ll u, ll v) {
        adj[u].pb(v);
        adj[v].pb(u);
    }

    // DFS to set depth + immediate parent
    void dfs(ll v, ll p) {
        par[v][0] = p;
        for (ll to : adj[v]) {
            if (to == p) continue;
            depth[to] = depth[v] + 1;
            dfs(to, v);
        }
    }

    // build LCA table
    void build(ll root = 0) {
        dfs(root, -1);
        for (ll j = 1; j < LOG; j++) {
            for (ll i = 0; i < n; i++) {
                if (par[i][j-1] != -1)
                    par[i][j] = par[par[i][j-1]][j-1];
            }
        }
    }

    // kth ancestor of node v
    ll kth_parent(ll v, ll k) {
        for (ll j = 0; j < LOG; j++) {
            if (k & (1LL << j)) {
                v = par[v][j];
                if (v == -1) break;
            }
        }
        return v;
    }

    // lowest common ancestor
    ll lca(ll a, ll b) {
        if (depth[a] < depth[b]) swap(a, b);

        // lift a to same depth
        a = kth_parent(a, depth[a] - depth[b]);

        if (a == b) return a;

        for (ll j = LOG - 1; j >= 0; j--) {
            if (par[a][j] != par[b][j]) {
                a = par[a][j];
                b = par[b][j];
            }
        }
        return par[a][0];
    }

    // distance between two nodes
    ll dist(ll a, ll b) {
        ll c = lca(a, b);
        return depth[a] + depth[b] - 2 * depth[c];
    }
};
