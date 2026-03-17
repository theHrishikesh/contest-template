
// topo sort (Kahn's Algorithm)
bool topo_sort(const vvll& dag)
{
    ll k = dag.size();
    vll indeg(k, 0);

    // compute indegrees
    rep(u,0,k)
    {
        each(v, dag[u])
        {
            indeg[v]++;
        }
    }

    queue<ll> q;
    rep(i,0,k)
    {
        if (indeg[i] == 0)
            q.push(i);
    }

    vll topo;

    while (!q.empty())
    {
        ll u = q.front(); q.pop();
        topo.pb(u);

        each(v, dag[u])
        {
            if (--indeg[v] == 0)
                q.push(v);
        }
    }

    // Optional safety check
    // if ((ll)topo.size() != k) -> cycle exists (should not happen for SCC DAG)

    return ((ll)topo.size() == k);
}

