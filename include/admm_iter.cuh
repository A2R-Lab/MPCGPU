#include <glass.cuh>


template <typename T>
compute_gamma(T * d_gamma, T * d_g, T * d_A, t *d_x, T* d_lambda, float rho, float sigma){
	/* gamma = g + sigma * x + A.T ( rho *x - lambda )*/
	
	/* launch it*/

}


template <typename T>
compute_gamma_kernel(){
	/* x_diff = rho * x - lambda */
	glass::axpby<T>(nx, rho, d_x, -1, lambda, d_xdiff);

	/* Atx = A.T * x_diff */
	glass::gemv<T, true>(nx, nx, 1, d_A, d_xdiff, d_Atx);

	/* gamma = g + sigma *x */
	glass::axpby<T>(nx, 1, d_g, sigma, x, d_gamma);

	/* gamma = Atx + gamma */
	glass::axpy<T>(nx, 1, d_Atx, d_gamma);
}


template <typename T>
update_z(T *d_A, T *d_x, T *d_lambda, T *d_z, float rho){
	/* z = clip( Ax + 1/rho * lambda )*/

	/* launch kernel*/

}

template <typename T>
update_z_kernel(T *d_A, T *d_x, T *d_lambda, T *d_z,  T * l, T * u, float rho){
	/* Ax = A * x */
	glass::gemv<T, false>(nx, nx, 1, d_A, d_x, d_Ax);

	/* z = Ax + 1/rho * lambda */
	glass::axpby<T>(nx, 1, d_Ax, 1/rho, lambda, d_z);

	/* z = clip(z) */
	glass::clip(nx, d_z, l, u);

}

template <typename T>
update_lambda(T * d_A, T * d_x, T * d_lambda, T * d_z, float rho, float l, float u){
	/* lambda = lambda + rho * (A * x - z )*/

	/* launch */
}

template <typename T>
update_lambda_kernel(T * d_A, T * d_x, T * d_lambda, T * d_z, float rho, float l, float u){
	/* Ax = A * x*/
	glass::gemv<T, false>(nx, nx, 1, d_A, d_x, d_Ax);

	/* Axz = Ax - z*/
	glass::axpby<T>(nx, 1, d_Ax, -1, d_z, d_Axz);

	/* lambda = lambda + rho * Axz */
	glass::axpy<T>(nx, rho, d_Axz, d_lambda);
}


template <typename T>
admm_iter(qp *prob, T *d_S, T *d_Pinv, T *d_x, T *d_lambda, T *d_z, float rho, float sigma){

	/*Allocate memory for gamma, */

	/*compute gamma*/
	compute_gamma(d_gamma, prob->d_g, prob->d_A, d_x, d_lambda, rho, sigma);

	/*call pcg*/
	solve_pcg(d_S, d_Pinv, d_x);

	/*update z*/
	update_z(d_A, d_x, d_lambda, d_z, rho, prob->l, prob->u);

	/*update lambda*/
	update_lambda(d_A, d_x, d_lambda, d_z, rho);
}