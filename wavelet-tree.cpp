
//Wavelet Tree
struct WaveletTree {
    long long lo, hi;
    WaveletTree *l, *r;
    vector<int> pref; // counts, indices stay int

    // Default constructor
    WaveletTree() : lo(0), hi(0), l(nullptr), r(nullptr) {}

    // Public constructor
    WaveletTree(const vector<long long> &data) : l(nullptr), r(nullptr) {
        if (data.empty()) {
            lo = hi = 0;
            return;
        }

        lo = *min_element(data.begin(), data.end());
        hi = *max_element(data.begin(), data.end());

        vector<long long> a = data; // reordered internally
        build(a.begin(), a.end(), lo, hi);
    }

private:
    void build(vector<long long>::iterator from,
            vector<long long>::iterator to,
            long long _lo, long long _hi) {

        lo = _lo;
        hi = _hi;
        l = r = nullptr;

        if (from >= to || lo == hi) return;

        long long mid = (lo + hi) >> 1;

        pref.reserve(to - from + 1);
        pref.push_back(0);
        for (auto it = from; it != to; ++it)
            pref.push_back(pref.back() + (*it <= mid));

        auto pivot = stable_partition(
            from, to,
            [mid](long long x) { return x <= mid; }
        );

        l = new WaveletTree();
        r = new WaveletTree();
        l->build(from, pivot, lo, mid);
        r->build(pivot, to, mid + 1, hi);
    }

    // INTERNAL: k is 0-based
    long long kth_internal(int lq, int rq, int k) const {
        if (lq >= rq) return -1;
        if (lo == hi) return lo;

        int leftCount = pref[rq] - pref[lq];
        if (k < leftCount)
            return l->kth_internal(pref[lq], pref[rq], k);
        else
            return r->kth_internal(
                lq - pref[lq],
                rq - pref[rq],
                k - leftCount
            );
    }

public:
    // PUBLIC: k is 1-based
    long long kth(int lq, int rq, int k) const {
        return kth_internal(lq, rq, k - 1);
    }

    // count occurrences of x
    int freq(int lq, int rq, long long x) const {
        if (lq >= rq || x < lo || x > hi) return 0;
        if (lo == hi) return rq - lq;

        long long mid = (lo + hi) >> 1;
        if (x <= mid)
            return l->freq(pref[lq], pref[rq], x);
        else
            return r->freq(
                lq - pref[lq],
                rq - pref[rq],
                x
            );
    }

    // count elements <= x
    int lte(int lq, int rq, long long x) const {
        if (lq >= rq || x < lo) return 0;
        if (hi <= x) return rq - lq;

        return l->lte(pref[lq], pref[rq], x)
            + r->lte(
                lq - pref[lq],
                rq - pref[rq],
                x
            );
    }

    // count elements in [low, high]
    int range_count(int lq, int rq, long long low, long long high) const {
        if (lq >= rq || high < lo || hi < low) return 0;
        if (low <= lo && hi <= high) return rq - lq;

        return l->range_count(pref[lq], pref[rq], low, high)
            + r->range_count(
                lq - pref[lq],
                rq - pref[rq],
                low, high
            );
    }
};
