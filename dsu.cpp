// DSU Algorithm
// Classic DSU (Union-Find)
struct DSU {
    vector<ll> parent, size;
    ll component_count;
    ll max_comp_size;

    DSU(ll n)
    {
        parent.resize(n);
        size.assign(n, 1);

        rep(i, n) parent[i] = i;

        component_count = n;
        max_comp_size = 1;
    }

    // find leader with path compression
    ll leader(ll x)
    {
        if (parent[x] == x) return x;
        return parent[x] = leader(parent[x]);
    }

    // merge two sets
    bool merge(ll x, ll y)
    {
        ll rx = leader(x);
        ll ry = leader(y);

        if (rx == ry) return false;

        // union by size
        if (size[rx] < size[ry]) swap(rx, ry);

        parent[ry] = rx;
        size[rx] += size[ry];

        component_count--;
        max_comp_size = max(max_comp_size, size[rx]);
        return true;
    }

    // are x and y in the same set?
    bool same(ll x, ll y)
    {
        return leader(x) == leader(y);
    }

    // size of the set containing x
    ll setsz(ll x)
    {
        return size[leader(x)];
    }
};
