vvll pref;

void build_pref(vvll &a, ll n, ll m)
{
    pref.assign(n + 1, vll(m + 1, 0));
    rep(i,1,n+1)
    {
        rep(j,1,m+1)
        {
            pref[i][j] = a[i-1][j-1]
                       + pref[i-1][j]
                       + pref[i][j-1]
                       - pref[i-1][j-1];
        }
    }
}

ll get(ll x1, ll y1, ll x2, ll y2)
{
    x1++, y1++, x2++, y2++;
    return pref[x2][y2]
         - pref[x1-1][y2]
         - pref[x2][y1-1]
         + pref[x1-1][y1-1];
}
