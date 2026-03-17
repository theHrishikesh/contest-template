/*Math Algorithms*/
// fast modular exponentiation
ll modpow(ll base, ll exp, ll mod)
{
    ll result = 1;
    base %= mod;
    while (exp > 0)
    {
        if (exp & 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}
ll modinv(ll a , ll mod) {
    return modpow(a, mod - 2,mod);
}
//floor div template
template <typename T>
T floor(T a, T b) {
return a / b - (a % b && (a ^ b) < 0);
}
//ceil div template
template <typename T>
T ceil(T x, T y) {
return floor(x + y - 1, y);
}
//balanced modulo or Euclidean modulo (out normal modulo)
template <typename T>
T bmod(T x, T y) {
return x - y * floor(x, y);
}
// gives a quotient reminder pair for a div
template <typename T>
pair<T, T> divmod(T x, T y) {
T q = floor(x, y);
return {q, x - q * y};
}
