
string to_base(ll a,ll b)
{
    if (b<2 || b>36) throw invalid_argument("base out of range");
    if (a == 0) return "0";
    static const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    string s;
    while (a > 0) {
        int rem = a % b;
        s.pb(digits[rem]);
        a /= b;
    }
    reverse(all(s));
    return s;
}
