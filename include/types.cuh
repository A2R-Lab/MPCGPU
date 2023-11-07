
template <typename T>
struct qp{
	T *d_H, *d_g, *d_A, *d_l , *d_u;
	int nx, nc;

	qp(T *d_H, 
		T *d_g, 
		T *d_A, 
		T *d_l, 
		T *d_u, 
		int nx, 
		int nc):
		d_H(d_H), d_g(d_g), d_A(d_A), d_l(d_l), d_u(d_u), nx(nx), nc(nc) {}
};