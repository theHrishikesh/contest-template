
/*Convex Hull*/

// Vector subtraction: b -> a
P sub(const P& a, const P& b) {
    return {a.first - b.first, a.second - b.second};
}

// Cross product of two vectors
ll cross_vec(const P& a, const P& b) {
    return a.first * b.second - a.second * b.first;
}

// Cross product (OA x OB)
ll cross(const P& O, const P& A, const P& B) {
    return cross_vec(sub(A, O), sub(B, O));
}

vpll convex_hull(vpll pts) {
    int n = pts.size();
    if (n <= 1) return pts;

    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());

    vpll hull;

    // Lower hull
    for (const auto& p : pts) {
        while (hull.size() >= 2 &&
            cross(hull[hull.size()-2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // Upper hull
    int lower_size = hull.size();
    for (int i = (int)pts.size() - 2; i >= 0; i--) {
        while ((int)hull.size() > lower_size &&
            cross(hull[hull.size()-2], hull.back(), pts[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(pts[i]);
    }

    hull.pop_back(); // remove duplicate start point
    return hull;
}
