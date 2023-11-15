
template <typename T>
struct qp{
	T *d_H, *d_g, *d_A, *d_Anorm, *d_l , *d_u;
	qp(T *d_H, 
		T *d_g, 
		T *d_A, 
		T *d_Anorm,
		T *d_l, 
		T *d_u):
		d_H(d_H), d_g(d_g), d_A(d_A), d_Anorm(d_Anorm), d_l(d_l), d_u(d_u) {}
};