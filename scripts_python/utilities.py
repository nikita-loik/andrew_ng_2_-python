


# UTILITY FUNCTIONS ===========================================================
def get_sigmoid(
        z):
    return 1 / (1 + np.exp(-z))


def get_hypothesis(β, X):  #1
#     return hypothesis vector h(n, 1), where n is n_samples
    return np.dot(X, β)


def get_hypothesis(β, X, n_samples, n_variables):  #2
    β = β.reshape(n_variables, -1)
    X = X.reshape(n_samples, -1)
    return get_sigmoid(np.dot(X, β))


def get_hypothesis(β, X):  #3
#     h(5000x1) = X(5000x401)*β(401x1)
    return get_sigmoid(np.dot(X, β[:, None]))


def get_hypothesis(β, X, n_samples, n_variables):  #5
    β = β.reshape(n_variables, -1)
    X = X.reshape(n_samples, -1)
#     return hypothesis vector h(n, 1), where n is n_samples
    return np.dot(X, β)



# 1 ===========================================================================
def cost_function(β, X, y):
    n_samples, n_variables = X.shape
#     hypothesis vector h(n, 1)
    h = get_hypothesis(β, X)
#     cost scalar J(1, 1)
    J = np.sum((y-h)**2)/(2*n_samples)
#     similarly, cost J can be calculated using dot-product
#     J = np.dot((y-h).T, y-h)/(2*n_samples)
#     technically, the result is an array (1,1) rather than a float
    return J


def get_gradient_descent(β, X, y, α, iterations):
    n_samples, n_variables = X.shape
    J_history = []
    for i in range(iterations):
#         hypothesis vector h(n, 1)
        h = get_hypothesis(β, X)
#         error vector e(n, 1)
        e = h - y
#         cost scalar J
        J = cost_function(β, X, y)
#         gradient vector g(k, 1)
        g = np.dot(X.T, e)/(n_samples)
#         updated β vector β(k, 1)
        β = β - α*g
#         updated J_history
        J_history += [J] 
    return β, J_history


# 2 ===========================================================================
# def cost_function(X, y, β):
def cost_function(β, X, y, n_samples, n_variables, λ=0.):
#     β = β.reshape(n_variables, -1)
#     X = X.reshape(n_samples, -1)
    y = y.reshape(n_samples, -1)
#     hypothesis vector h(n, 1)
    h = get_hypothesis(β, X, n_samples, n_variables)
#     cost scalar J(1, 1)
    J = (- np.dot(y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h)))/n_samples
#     similarly cost J can be calculated using np.multiply together with np.sum
#     cost = -np.sum(np.multiply(y, np.log(h)) + np.multiply((1-y), np.log(1-h)))/n_samples
#     regularisation scalar (R)
    R = λ*np.dot(β[1:].T,β[1:])/(2*n_samples)
    return (J + R)[0][0]

def optimise_β_1 (β, X, y, n_samples, n_variables, λ=0.):
    return optimize.fmin(cost_function, x0=β, args=(X, y, n_samples, n_variables, λ), maxiter=1500, full_output=True)

def get_prediction(β, X, n_samples, n_variables):
    return get_hypothesis(β, X, n_samples, n_variables) >= 0.5


# 3 ===========================================================================
def cost_function(β, X, y, λ = 0.):
    n_samples, n_variables = X.shape  # X(5000x401)
#     hypothesis vector h(5000x1) = X(5000x401)*β(401x1)
    h = get_hypothesis(β, X)
#     cost function scalar J (1x1) = y.T(1x5000)*h(5000x1)
    J = (- np.dot(y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h)))/n_samples
#     regularisation scalar (R)
    R = λ*np.dot(β[1:].T,β[1:])/(2*n_samples)
#     return (cost.item(0))
    return (J + R)[0,0]


def get_gradient(β, X, y, λ=0.):
    n_samples, n_variables = X.shape
#     hypothesis (h)
    h = get_hypothesis(β, X)
#     error vector e(5000x1) = h(5000x1) - y(5000x1)
    e = h - y
#     gradient vector g(401x1) = e.T(1x5000)*X(401x5000)
    g = np.dot(X.T,e)/(n_samples)
#     regularisation term vector (r(400x1)) — derivative of the regularisation term of the cost funtion
    r = β[1:, None]*(λ/n_samples)
    g[1:] = g[1:] + r
    return g.flatten()


# 4 ===========================================================================



def forward_propagation(β_flat, layer, X_flat, n_samples):
    '''Forward Propagation is the hypothesis function for Neural Networks'''
    β_set = reshape_β(β_flat, layer)
#     H_0 (5000, 400)
    H = X_flat.reshape(n_samples, -1)
#     Z_H = ()
    H_byLayer = ()
    for β in β_set:
#         print(H.shape)
#         Z_l (5000, k_l); l is the number of layers [0, ...,l]; k is the number of neurons in a layer l [1,...,k]
        Z = np.dot(np.insert(H, 0, 1, axis=1), β.T)
#         H_l (5000, k_l); l is the number of layers [0, ...,l]; k is the number of neurons in a layer l [1,...,k]
        H = get_sigmoid(Z)
#         Z_H += ((Z, H),)
        H_byLayer += (H,)
#     H_2 (5000, 10)
    return H_byLayer

def get_sigmoid_gradient(Z):
    return get_sigmoid(Z)*(1-get_sigmoid(Z))


def cost_function(β_flat, layer, X_flat, n_samples, y, yUnique, λ = 0.):
    X = X_flat.reshape(n_samples, -1)
    Y = np.array([yUnique]* y.shape[0]) == y
    β_set = reshape_β(β_flat, layer)
    J = 0
    for n in range(n_samples):
        x_n = X[n:n+1,:]
        y_n = Y[n:n+1,:]
#         hypothesis vector h_n(1, 10)
        h_n = forward_propagation(β_flat, layer, x_n, 1)[len(β_set)-1]
#         cost function scalar j_n(1, 1) = y_n(1, 10)*h_n.T(10, 1)
        j_n = (- np.dot(y_n, np.log(h_n).T) - np.dot((1-y_n), np.log(1-h_n).T))
        J += j_n
#     regularisation term (R)
    cummulativeR = 0
    for β in β_set:
        cummulativeR += np.sum(β*β) #element-wise multiplication
    cummulativeR *= λ/(2*n_samples)
    return J[0][0]/n_samples + cummulativeR

    def back_propagation(β_flat, layer, X_flat, n_samples, y, yUnique):
    Y = np.array([yUnique]* y.shape[0]) == y
    β_set = reshape_β(β_flat, layer)

    deltaSet = ()
#     hypothesis matrix E(5000, 10)
    H = forward_propagation(β_flat, layer, X_flat, n_samples)
#     error matrix E(5000, 10)
    E = H[len(layer)-2] - Y
    for l in reversed(range(len(layer)-1)):
        E = np.dot(E*get_sigmoid_gradient(H[l]), β_set[l])[:,1:]
        deltaSet = (np.dot(H[l].T, np.insert(E, 0, 1, axis=1)),) + deltaSet
    flatDelta = flatten_β(deltaSet)
    return β_flat + flatDelta/n_samples


# 5 ===========================================================================
def cost_function(β, X, y, n_samples, n_variables, λ=0.):
    β = β.reshape(n_variables, -1)
    X = X.reshape(n_samples, -1)
    y = y.reshape(n_samples, -1)
#     hypothesis vector h(n, 1)
    h = get_hypothesis(β, X, n_samples, n_variables)
#     cost scalar J(1, 1); technically the result is an array (1,1) rather than a float
    J = np.dot((y-h).T, y-h)/(2*n_samples)
#     similarly cost J can be calculated using np.sum
#     J = np.sum((y-h)**2)/(2*n_samples)
    R = λ*np.dot(β.T, β)/(2*n_samples)
    return (J + R)[0][0]

def get_gradient(β, X, y, n_samples, n_variables, λ=0.):
    β = β.reshape(n_variables, -1)
    X = X.reshape(n_samples, -1)
    y = y.reshape(n_samples, -1)
#     hypothesis vector h(n, 1)
    h = get_hypothesis(β, X, n_samples, n_variables)
#     error vector e(n, 1) = h(n, 1) - y(n, 1)
    e = h - y
#     gradient vector g(k, 1) = X(n, k).T*e(n, 1)*
    g = np.dot(X.T,e)/(n_samples)
#     regularisation term vector (r(400x1)) — derivative of the regularisation term of the cost funtion
    r = β[1:]*(λ/n_samples)
    g[1:] = g[1:] + r
    return g.flatten()

def plot_regression(β, X, y, n_samples, n_variables):  
    β = β.reshape(n_variables, -1)
    X = X.reshape(n_samples, -1)
    y = y.reshape(n_samples, -1)
    
    y_fit = np.dot(X, β)
    
    MSE = np.sum((y - y_fit)**2)/y.shape[0]
    
    plt.plot(X[:,1:], y, 'o', X[:,1:], y_fit, '-')
    plt.xlabel("X")
    plt.ylabel("Y")
    print ("β_0:", β[0][0],
           "\nβ_1:", β[1][0],
           "\nRegression: Y =", '{:10.2f}'.format(β[0][0]), '+', '{:10.2f}'.format(β[1][0]), "X"
           "\nMSE =",'{:10.2f}'.format(MSE))
    return plt.show()


