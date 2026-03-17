/*modint template */
template<ll M>
struct modint {
    ll v;
    constexpr modint(ll _v = 0) noexcept {
        v = _v % M;
        if (v < 0) v += M;
    }
    // addition / subtraction
    constexpr modint& operator+=(modint o) noexcept {
        v += o.v;
        if (v >= M) v -= M;
        return *this;
    }
    constexpr modint& operator-=(modint o) noexcept {
        v -= o.v;
        if (v < 0) v += M;
        return *this;
    }
    // multiplication
    constexpr modint& operator*=(modint o) noexcept {
        v = (unsigned long long)v * o.v % M;
        return *this;
    }
    // binary exponentiation
    friend modint power(modint x, ll e) {
        modint res(1);
        while (e > 0) {
            if (e & 1) res *= x;
            x *= x;
            e >>= 1;
        }
        return res;
    }
    // inverse (M must be prime)
    friend modint inv(modint x) {
        return power(x, M - 2);
    }
    // division
    constexpr modint& operator/=(modint o) noexcept {
        return *this *= inv(o);
    }

    // lee-way operators
    constexpr modint operator+(modint o) const noexcept { return modint(*this) += o; }
    constexpr modint operator-(modint o) const noexcept { return modint(*this) -= o; }
    constexpr modint operator*(modint o) const noexcept { return modint(*this) *= o; }
    constexpr modint operator/(modint o) const noexcept { return modint(*this) /= o; }

    // I/O
    friend ostream& operator<<(ostream& os, modint x) {
        return os << x.v;
    }
    friend istream& operator>>(istream& is, modint& x) {
        ll t; is >> t; x = modint(t); return is;
    }
};

using mint = modint<MODe>;

/*Binomial Templates*/
vector<mint> fact, invfact;
void init_nCr(ll N) {
    fact.resize(N + 1);
    invfact.resize(N + 1);

    fact[0] = 1;
    for (ll i = 1; i <= N; i++)
        fact[i] = fact[i - 1] * i;

    invfact[N] = inv(fact[N]);
    for (ll i = N; i > 0; i--)
        invfact[i - 1] = invfact[i] * i;
}
mint nCr(ll n, ll r) {
    if (r < 0 || r > n) return 0;
    return fact[n] * invfact[r] * invfact[n - r];
}
