#include <admm_outer.cuh>
#include <limits>

using namespace std;

int main(){
	/*
	H = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
	g = np.array([[-8], [-3], [-3]])
	A = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
	l = np.array([[3], [0], [-10], [-10], [-10]])
	u = np.array([[3], [0], [np.inf], [np.inf], [np.inf]])
	*/
	float H[9] = {6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0};
	float g[3] = {-8.0, -3.0, -3.0};
	float A[15] = {1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
	float l[5] = {3.0, 0.0, -10.0, -10.0, -10.0};
	float u[5] = {3.0, 0.0, std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max()};

	float x[3] = {0.0, 0.0, 0.0};
	float z[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
	float lambda[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
	float rho = 0.1;

	admm_solve_outer(H, g, A, l, u, x, lambda, z, rho);
	
	std::cout << "X: ";
	for(int i=0; i<3; i++){
		std::cout << x[i] << " ";
	}
	std::cout << "\n\n";
	std::cout << "lambda: ";
	for(int i=0; i<5; i++){
		std::cout << lambda[i] << " ";
	}

	std::cout << "\n\n";
	std::cout << "z: ";
	for(int i=0; i<5; i++){
		std::cout << z[i] << " ";
	}

	std::cout << "\n\n";



}