
// Tarjans Algorithm for SCC
struct SCC
{
    ll n,timer=0,compcnt=0;
    vvll g;
    vll disc,low,comp;
    stkll stk;
    vbl instack;
    
    SCC(ll n) : n(n), g(n), disc(n,-1), low(n), comp(n,-1), instack(n,false){}
    
    void add_edge(ll u,ll v)
    {
        g[u].pb(v);
    }
    
    void dfs(ll u)
    {
        disc[u] = low[u] = ++timer;
        stk.push(u);
        instack[u]=true;
        
        each(v,g[u])
        {
            if (disc[v]==-1)
            {
                dfs(v);
                chmin(low[u],low[v]);
            }
            elif (instack[v])
            {
                chmin(low[u],disc[v]);
            }
        }
        
        if (low[u]==disc[u])
        {
            while(true)
            {
                ll v = stk.top();stk.pop();
                instack[v] = false;
                comp[v] = compcnt;
                if (v==u) break;
            }
            compcnt++;
        }
    }
    
    void run()
    {
        rep(i,0,n)
        {
            if (disc[i]==-1) dfs(i);
        }
    }
};
