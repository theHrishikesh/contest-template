vll primes_upto(ll n)
{
    vbl is_prime(n + 1, true);
    vll primes;

    is_prime[0] = is_prime[1] = false;

    for (ll i = 2; i * i <= n; i++)
    {
        if (is_prime[i])
        {
            for (ll j = i * i; j <= n; j += i)
                is_prime[j] = false;
        }
    }

    rep(i, 2, n + 1)
        if (is_prime[i])
            primes.push_back(i);

    return primes;
}
