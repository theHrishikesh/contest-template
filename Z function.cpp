auto z_function = [](string s)
{
    vll z(len(s),0);

    ll l = 0, r = 0;
    rep(i,1,len(s))
    {
        if (i <= r)
        {
            z[i] = min(z[i - l], r - i + 1);
        }

        while (i + z[i] < len(s) && s[z[i]] == s[i + z[i]])
        {
            z[i]++;
        }

        if (i + z[i] - 1 > r)
        {
            l = i;
            r = i + z[i] - 1;
        }
    }

    return z;
};
