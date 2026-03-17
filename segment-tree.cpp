//Segment Tree
struct SegTree {
    ll n;
    vll tree;

    SegTree(ll _n) {
        n = _n;
        tree.assign(4*n, 0);
    }


    void build(ll node, ll l, ll r, vll &a) {
        if (l == r) {
            tree[node] = a[l];
            return;
        }
        ll mid = (l + r) / 2;
        build(2*node, l, mid, a);
        build(2*node+1, mid+1, r, a);
        tree[node] = tree[2*node] + tree[2*node+1];
    }


    ll query(ll node, ll l, ll r, ll ql, ll qr) {
        if (qr < l || r < ql) return 0;              
        if (ql <= l && r <= qr) return tree[node];  
        ll mid = (l + r) / 2;
        return query(2*node, l, mid, ql, qr)
            + query(2*node+1, mid+1, r, ql, qr);
    }


    void update(ll node, ll l, ll r, ll pos, ll val) {
        if (l == r) {
            tree[node] = val;
            return;
        }
        ll mid = (l + r) / 2;
        if (pos <= mid) update(2*node, l, mid, pos, val);
        else update(2*node+1, mid+1, r, pos, val);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
};
