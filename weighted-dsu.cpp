// Weighted DSU Algorithm
struct WDSU {
    vector<ll> parent, size;
    vector<ll> diff_weight;   // weight[x] - weight[parent[x]]
    ll component_count;
    ll max_comp_size;

    WDSU(ll n)
    {
        parent.resize(n);
        size.assign(n, 1);
        diff_weight.assign(n, 0);  // initialize all weights to 0

        rep(i,n) parent[i] = i;

        component_count = n;
        max_comp_size = 1;
    }

    // path compression + accumulate weights
    ll leader(ll x)
    {
        if (parent[x] == x) return x;
        ll p = leader(parent[x]);
        diff_weight[x] += diff_weight[parent[x]];
        return parent[x] = p;
    }

    // weight(x) = value[x] - value[root(x)]
    ll weight(ll x)
    {
        leader(x);
        return diff_weight[x];
    }

    // merge with constraint: value[y] - value[x] = w
    bool merge(ll x, ll y, ll w)
    {
        ll rx = leader(x);
        ll ry = leader(y);

        ll wx = weight(x); // weight from x to rx
        ll wy = weight(y); // weight from y to ry

        if (rx == ry) return false;

        // Solve: value[y] - value[x] = w
        // → (wy + diff_weight[ry-root]) - (wx + diff_weight[rx-root]) = w
        // Since roots have diff_weight = 0:
        ll diff = w + wx - wy;

        // union by size
        if (size[rx] < size[ry]) {
            swap(rx, ry);
            diff = -diff;
        }

        parent[ry] = rx;
        diff_weight[ry] = diff;    // weight[ry] = value[ry] - value[rx]
        size[rx] += size[ry];

        component_count--;
        max_comp_size = max(max_comp_size, size[rx]);
        return true;
    }

    // same component?
    bool same(ll x, ll y)
    {
        return leader(x) == leader(y);
    }

    // difference value[y] - value[x]
    ll diff(ll x, ll y)
    {
        return weight(y) - weight(x);
    }

    ll setsz(ll x)
    {
        return size[leader(x)];
    }
};
