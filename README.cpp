#include <bits/stdc++.h>
using namespace std;
struct IoSetup {
    IoSetup() {
        cin.tie(nullptr);
        ios::sync_with_stdio(false);
        cout << fixed << setprecision(15);
        cerr << fixed << setprecision(15);
    }
} iosetup;
void setIO(string s)
{
freopen((s + ".in").c_str(), "r", stdin);
freopen((s + ".out").c_str(), "w", stdout);
}
#define overload5(_1,_2,_3,_4,_5,name,...) name
#define overload4(_1,_2,_3,_4,name,...) name
#define overload3(_1,_2,_3,name,...) name
#define rep1(n) for(ll i=0;i<n;++i)
#define rep2(i,n) for(ll i=0;i<n;++i)
#define rep3(i,a,b) for(ll i=a;i<b;++i)
#define rep4(i,a,b,c) for(ll i=a;i<b;i+=c)
#define rep(...) overload4(__VA_ARGS__,rep4,rep3,rep2,rep1)(__VA_ARGS__)
#define rrep1(n) for(ll i=n;i--;)
#define rrep2(i,n) for(ll i=n;i--;)
#define rrep3(i,a,b) for(ll i=b;i-->(a);)
#define rrep4(i,a,b,c) for(ll i=(a)+((b)-(a)-1)/(c)*(c);i>=(a);i-=c)
#define repsq(i, n) for (ll i = 1; ((i) * (i) < n); ++i)
#define rrep(...) overload4(__VA_ARGS__,rrep4,rrep3,rrep2,rrep1)(__VA_ARGS__)
#define each1(i,a) for(auto&&i:a)
#define each2(x,y,a) for(auto&&[x,y]:a)
#define each3(x,y,z,a) for(auto&&[x,y,z]:a)
#define each4(w,x,y,z,a) for(auto&&[w,x,y,z]:a)
#define each(...) overload5(__VA_ARGS__,each4,each3,each2,each1)(__VA_ARGS__)
#define all1(i) begin(i),end(i)
#define all2(i,a) begin(i),begin(i)+a
#define all3(i,a,b) begin(i)+a,begin(i)+b
#define all(...) overload3(__VA_ARGS__,all3,all2,all1)(__VA_ARGS__)
#define rall1(i) rbegin(i),rend(i)
#define rall2(i,a) rbegin(i),rbegin(i)+a
#define rall3(i,a,b) rbegin(i)+a,rbegin(i)+b
#define rall(...) overload3(__VA_ARGS__,rall3,rall2,rall1)(__VA_ARGS__)
#define len(x) (ll)(x).size()
#define sum(...) accumulate(all(__VA_ARGS__),0LL)
#define elif else if
#define pb push_back
#define pf push_front
#define eb emplace_back
#define Test int testing; cin >> testing; while(testing--)
#define dbg(...) cout << #__VA_ARGS__ << " = ", _print(__VA_ARGS__)
#define sint(...) int __VA_ARGS__; in(__VA_ARGS__)
#define sll(...) ll __VA_ARGS__; in(__VA_ARGS__)
#define sstr(...) string __VA_ARGS__; in(__VA_ARGS__)
#define sch(...) char __VA_ARGS__; in(__VA_ARGS__)
#define sdbl(...) double __VA_ARGS__; in(__VA_ARGS__)
#define sld(...) ld __VA_ARGS__; in(__VA_ARGS__)
#define svll(n, v) vll v(n); scan(v)
#define svii(n, v) vii v(n); scan(v)
#define svec(type, n, v) vector<type> v(n); scan(v)
#define s1vll(n,v) vll v(n + 1); rep(i, 1, n + 1) cin >> v[i];
#define s1vii(n,v) vii v(n + 1); rep(i, 1, n + 1) cin >> v[i];
#define fi first
#define se second
// ----------------------------------------------------------------------------------------
//binarySearch macros
// ----------------------------------------------------------------------------------------
#define lb(v,target) lower_bound(all(v), target)
#define rlb(v,target) lower_bound(rall(v), target)
#define lbset(s,target) s.lower_bound(target)
#define ub(v,target) upper_bound(all(v), target)
#define lbidx(v,target) lb(v,target)-v.begin()
#define ubidx(v,target) ub(v,target)-v.begin()
#define rub(v,target) upper_bound(rall(v), target) //Equivalent to finding the largest element smaller than or equal to target
#define ubset(s,target) s.upper_bound(target)
// ----------------------------------------------------------------------------------------
#define digitcount(n) ((n) == 0 ? 1 : (int)log10(abs(n)) + 1)
// ----------------------------------------------------------------------------------------
//string macros
#define str(x) to_string(x)
// ----------------------------------------------------------------------------------------
const int dx4[4] = {1, 0, -1, 0};
const int dy4[4] = {0, 1, 0, -1};
const int dx8[8] = {1, 1, 0, -1, -1, -1, 0, 1};
const int dy8[8] = {0, 1, 1, 1, 0, -1, -1, -1};
// ----------------------------------------------------------------------------------------
// universal shorthands
template<class T> using V = vector<T>;                  // dynamic array
template<class T, size_t N> using A = array<T, N>;      // fixed-size array
// *** Primitive short aliases ***
using ll = long long;
using ld = long double;
using ull = unsigned long long;
// *** Container/type shortcuts ***
using vch = vector<char>;
using vvch = vector<vch>;
using vvcc = vector<vch>;
using vll = vector<ll>;
using vvll = vector<vll>;
using vii = vector<int>;
using vvii = vector<vii>;
using vecs = vector<string>;
// *** Pair shortcuts ***
using P = pair<ll,ll>;
using pll = pair<ll,ll>;
using pdd = pair<ld,ld>;
using pii = pair<int,int>;
// *** Vector of pairs ***
using vpii = vector<pii>;
using vvpii = vector<vpii>;
using vpll = vector<pll>;
using vvpll = vector<vpll>;
using vpci = vector<pair<char,int>>;
using vpcl = vector<pair<char,ll>>;
// *** Boolean and set containers ***
using vbl = vector<bool>;
using vvbl = vector<vbl>;
using usetii = unordered_set<int>;
using usetll = unordered_set<ll>;
using setii = set<int>;
using setll = set<ll>;
using setstr = set<string>;
using usetpll = unordered_set<pll>;
using usetpii = unordered_set<pii>;
// *** Stack shortcuts ***
using stkint = stack<int>;
using stkll = stack<ll>;
using stkpii = stack<pii>;
using stkpll = stack<pll>;
static constexpr ll MOD9 = 998244353;
static constexpr ll MODe = 1000000007;
// ----------------------------------------------------------------------------------------
template<class T> auto vmin(const T& a){ return *min_element(all(a)); }
template<class T> auto vmax(const T& a){ return *max_element(all(a)); }
template<class T, class U> bool chmin(T& a, const U& b){ if(a > T(b)){ a = b; return 1; } return 0; }
template<class T, class U> bool chmax(T& a, const U& b){ if(a < T(b)){ a = b; return 1; } return 0; }
template<typename T>
using maxpq = priority_queue<T>;
template<typename T>
using minpq = priority_queue<T, vector<T>, greater<T>>;
//scan
inline void scan() {}
inline void scan(int &a) { std::cin >> a; }
inline void scan(unsigned &a) { std::cin >> a; }
inline void scan(long &a) { std::cin >> a; }
inline void scan(long long &a) { std::cin >> a; }
inline void scan(unsigned long long &a) { std::cin >> a; }
inline void scan(char &a) { std::cin >> a; }
inline void scan(float &a) { std::cin >> a; }
inline void scan(double &a) { std::cin >> a; }
inline void scan(long double &a) { std::cin >> a; }
inline void scan(std::vector<bool> &vec) {for (size_t i = 0; i < vec.size(); i++) { int a;scan(a);vec[i] = a;}}
inline void scan(std::string &a) { std::cin >> a; }
template <class T>
inline void scan(std::vector<T> &vec);
template <class T, size_t size>
inline void scan(std::array<T, size> &vec);
template <class T, class L>
inline void scan(std::pair<T, L> &p);
template <class T, size_t size>
inline void scan(T (&vec)[size]);
template <class T>
inline void scan(std::vector<T> &vec) {for (auto &i : vec) scan(i);}
template <class T>
inline void scan(std::deque<T> &vec) {for (auto &i : vec) scan(i);}
template <class T, size_t size>
inline void scan(std::array<T, size> &vec) {for (auto &i : vec) scan(i);}
template <class T, class L>
inline void scan(std::pair<T, L> &p) {scan(p.first);scan(p.second);}
template <class T, size_t size>
inline void scan(T (&vec)[size]) {for (auto &i : vec) scan(i);}
template <class T>
inline void scan(T &a) {std::cin >> a;}
inline void in() {}
template <class Head, class... Tail>
inline void in(Head &head, Tail &...tail) {
    scan(head);
    in(tail...);
}
//print functions
inline void print(const bool &a) { std::cout << a; }
inline void print(const int &a) { std::cout << a; }
inline void print(const unsigned &a) { std::cout << a; }
inline void print(const long &a) { std::cout << a; }
inline void print(const long long &a) { std::cout << a; }
inline void print(const unsigned long long &a) { std::cout << a; }
inline void print(const char &a) { std::cout << a; }
inline void print(const char a[]) { std::cout << a; }
inline void print(const float &a) { std::cout << a; }
inline void print(const double &a) { std::cout << a; }
inline void print(const long double &a) { std::cout << a; }
inline void print(const std::string &a) {for (auto &&i : a) print(i);}
template <class T>
inline void print(const std::vector<T> &vec);
template <class T, size_t size>
inline void print(const std::array<T, size> &vec);
template <class T, class L>
inline void print(const std::pair<T, L> &p);
template <class T, size_t size>
inline void print(const T (&vec)[size]);
template <class T>
inline void print(const std::vector<T> &vec) {
    if (vec.empty()) return;
    print(vec[0]);
    for (auto i = vec.begin(); ++i != vec.end();) {
        std::cout << ' ';
        print(*i);
    }
}
template <typename T>
inline void print(const std::set<T>& s) {
    if (s.empty()) return;
    auto it = s.begin();
    print(*it);
    for (++it; it != s.end(); ++it) {
        cout << ' ';
        print(*it);
    }
}
template <class T>
inline void print(const std::deque<T> &vec) {
    if (vec.empty()) return;
    print(vec[0]);
    for (auto i = vec.begin(); ++i != vec.end();) {
        std::cout << ' ';
        print(*i);
    }
}
template <class T, size_t size>
inline void print(const std::array<T, size> &vec) {
    print(vec[0]);
    for (auto i = vec.begin(); ++i != vec.end();) {
        std::cout << ' ';
        print(*i);
    }
}
template <class T, class L>
inline void print(const std::pair<T, L> &p) {
    print(p.first);
    std::cout << ' ';
    print(p.second);
}
template <class T, size_t size>
inline void print(const T (&vec)[size]) {
    print(vec[0]);
    for (auto i = vec; ++i != end(vec);) {
        std::cout << ' ';
        print(*i);
    }
}
template <class T>
inline void print(const T &a) {
    std::cout << a;
}
inline void out() { std::cout << '\n'; }
template <class T>
inline void out(const T &t) {
    print(t);
    std::cout << '\n';
}
template <class Head, class... Tail>
inline void out(const Head &head, const Tail &... tail) {
    print(head);
    if constexpr (sizeof...(tail) > 0) {
        std::cout << ' ';
        out(tail...);
    } else {
        std::cout << '\n';
    }
}

template<typename T>
void __print(const T &x) { cout << x; }

template<typename T, typename V>
void __print(const pair<T, V> &x) {
    cout << "{"; __print(x.first); cout << ", "; __print(x.second); cout << "}";
}
template<typename T>
void __print(const vector<T> &v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        __print(v[i]);
        if (i + 1 != v.size()) cout << ", ";
    }
    cout << "]";
}
template<typename T>
void __print(const deque<T> &v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        __print(v[i]);
        if (i + 1 != v.size()) cout << ", ";
    }
    cout << "]";
}

template<typename T>
void __print(const set<T> &v) {
    cout << "{";
    bool first = true;
    for (auto &i : v) {
        if (!first) cout << ", ";
        __print(i);
        first = false;
    }
    cout << "}";
}
template<typename T>
void __print(const multiset<T> &v) { __print(set<T>(v.begin(), v.end())); }

template<typename K, typename V>
void __print(const map<K, V> &v) {
    cout << "{";
    bool first = true;
    for (auto &i : v) {
        if (!first) cout << ", ";
        __print(i);
        first = false;
    }
    cout << "}";
}
template<typename T>
void __print(const unordered_set<T> &v) { __print(set<T>(v.begin(), v.end())); }
template<typename K, typename V>
void __print(const unordered_map<K, V> &v) { __print(map<K, V>(v.begin(), v.end())); }
void _print() { cout << endl; }
template<typename T, typename... V>
void _print(T t, V... v) {
    __print(t);
    if (sizeof...(v)) cout << ", ";
    _print(v...);
}
template<typename... Args>
void __print(const std::tuple<Args...>& t) {
    cout << "(";
    std::apply([](const auto&... args) {
        size_t i = 0;
        ((cout << (i++ ? ", " : "") << args), ...);
    }, t);
    cout << ")";
}
template<typename T>
void print_stack(std::stack<T> s)
{
    std::cout << "[ ";
    while (!s.empty())
    {
        std::cout << s.top() << " ";
        s.pop();
    }
    std::cout << "]\n";
}
template<typename T>
void print_queue(std::queue<T> q)
{
    std::cout << "[ ";
    while (!q.empty())
    {
        std::cout << q.front() << " ";
        q.pop();
    }
    std::cout << "]\n";
}
void print_pq(minpq<pll> pq)
{
    cout << "pq: ";
    while (!pq.empty())
    {
        auto x = pq.top(); pq.pop();
        cout << "(" << x.fi << "," << x.se << ") ";
    }
    cout << "\n";
}
void print_dq(deque<pll> dq)
{
    cout << "dq: ";
    for (auto &x : dq)
    {
        cout << "(" << x.fi << "," << x.se << ") ";
    }
    cout << "\n";
}
int main()
{
}



