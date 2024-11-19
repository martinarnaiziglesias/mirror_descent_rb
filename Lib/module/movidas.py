import numpy as np
from scipy.stats import norm
from scipy.stats import t
from scipy.optimize import minimize
from math import log10, floor, ceil
import student_mixture as sm
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
import ipywidgets as widgets

#we have the random seed in order to replicate the results from the numerical section. In practice we shluld erase this
np.random.seed(0)

def round_to_1(x):
    return round(x, -int(floor(log10(np.abs(x)))))


def is_positivesemidefinite(m, tol=1e-8):
    e = np.linalg.eigvalsh(m)
    return np.all(e > -tol)


def cov2corr(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def car_func(y,x,a):
    y = y*x
    z = - np.sum(y)        
    z = z - a 
    return 1 if z > 0 else 0

def pos_func(y,x,a):
    y = y*x
    z = - np.sum(y)        
    z = z - a 
    return z if z > 0 else 0


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return floor(n * multiplier) / multiplier


def sample_expected_shortfall(x, alpha):
    return -np.mean(np.sort(x, axis=0)[:int(len(x) * (1 - alpha))], axis=0)


def sample_value_at_risk(x, alpha):
    return np.quantile(-x, alpha)

def fit_student_mixture(X, n_sm = 2):
    ### Fit Student-t mixture (2 components) to chosen stock returns
    SM_model = sm.StudentMixture(n_components=n_sm).fit(X)
    
    locs = SM_model.locations_
    scales = SM_model.scales_
    probs = SM_model.weights_
    dofs = SM_model.dofs_
    
    ### Some adjustments on the parameters of the final model 
    locs = np.array([round_to_1(i) for i in locs.flatten()]).reshape(locs.shape)
    scales = np.array([round_to_1(i) for i in scales.flatten()]).reshape(scales.shape)
    
    dofs[0] = round_down(dofs[0], 1)
    dofs[1] = round_down(dofs[1], 1)
    
    probs[0] = round_up(probs[0], 1)
    probs[1] = 1-probs[0]
    
    
    SM_model.locations_ = locs
    SM_model.scales_ = scales
    SM_model.weights_ = probs
    SM_model.dofs_ = dofs

    return SM_model

def get_df(assets):
    df_all = pd.concat([pd.read_excel('Data/SP_RC.xlsx',index_col=0, header=[0,1]), pd.read_excel('Data/SP_RC2.xlsx',index_col=0, header=[0,1])])
    df_all = df_all.dropna(how='all').dropna(axis=1)
    df_all = df_all.pct_change().dropna()
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    df = df_all[assets]
    return df



def regression_plot(x, y, dist=1):
    """
    Performs a regression of Y on X, taking elements of X every 'dist' steps,
    e.g., 10,000, and plots the regression line as well as showing the slope value.

    Args:
    x (array): independent variable
    y (array): dependent variable
    dist (int): distance between selected points

    Returns:
    float: The slope of the regression line
    """
    
    # Filter X and Y, taking only every "dist" elements
    y_orig=y
    x_orig=x
    y = y[np.arange(len(y)) % dist == 0]
    x = x[np.arange(len(x)) % dist == 0]
    
    # Ensure both arrays are reshaped to 2D for the model
    y = y.reshape(-1, 1)
    x = x.reshape(-1, 1)
    
    # Create and fit the regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # Get the slope and intercept (coefficient and intercept)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    # Plot the scatter of the filtered points
    plt.scatter(x_orig, y_orig, s=0.1, label="Data")
    
    # Plot the regression line
    plt.plot(x, slope * x + intercept, color='red', label=f'Line: y = {slope:.2f}x + {intercept:.2f}')
    
    # Add the equation of the line as text in the plot
    equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(x.min(), y.max(), equation_text, fontsize=12, color='red')
    
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # Show the plot
    plt.show()
    
    # Return the slope as the result
    return slope


class stoch_sol:
    def __init__ (self, X, M, n_val, nb_asset, budgets,  n_epochs = 10, gamma=1, exp = 0.65, beta = 0.95, differences= False, gamma_star=None):
        if differences==True:
            if gamma_star is None:
                raise ValueError("gamma_star must be provided if differences is True")
            self.gamma_star = gamma_star
        ones = np.ones(nb_asset)
        xi = 0
        k = 1
        c = 0
        ys = np.zeros((nb_asset, n_val*n_epochs))
        xi_final=[]
        c_final = []
        y_bar= np.zeros((1, nb_asset))
        y_bar_hist=[]
        xi_bar=0
        xi_bar_hist = []
        y = budgets / np.std(X, axis=0)
        
        for j in tqdm(range(n_epochs)):
            np.random.shuffle(X)
            for i in range(n_val):    
                stepsize = (gamma /k)** exp
                y_min = min(min(y),1)
                Gamma_grad =  - (X[i] * ones * car_func(y,X[i],xi) / (ones - beta * ones)) - budgets/y 
                c = c - 1/(k+1)*(c-(xi + 1/(1-beta)*pos_func(y,X[i],xi)))
                xi = xi - stepsize*(1 - (car_func(y,X[i],xi) / (1 - beta)))
                y = y*np.exp(-stepsize*y_min*Gamma_grad)
                sum_y = np.sum(y)
                if sum_y>M:
                    y = M/sum_y*y
                for asset_index in range(nb_asset):
                    ys[asset_index, j*n_val + i] = y[asset_index] 
                xi_final.append(xi)
                c_final.append(c)
                if differences == True:
                    y_bar= y_bar * i/(i+1) + y *  1/(i+1)
                    xi_bar = xi_bar * i/(i+1) + xi *  1/(i+1)
                    y_bar_hist.append(y_bar)
                    xi_bar_hist.append(xi)
                k=k+1
        self.ys = ys
        self.xi = xi
        self.c = c
        self.y = y
        self.xi_final = xi_final
        self.c_final = c_final
        self.nb_asset = nb_asset
        self.beta = beta

        if differences == True:
            differences = []
            for i in tqdm(range(len(xi_bar_hist))):
                #DOUBT HERE
                differences.append(np.mean(xi_bar_hist[i] + 1/(1-beta) * np.maximum(-np.dot(y_bar_hist[i][0],X.T) - xi_bar_hist[i],0) - np.sum(budgets* np.log(y_bar_hist[i][0]))) - gamma_star)
            self.differences = differences

    def results(self, real_sol=None, real_ES = None, real_VaR = None):
        if real_sol is not None:
            print('==== True Risk Budgeting portfolio ====')
            print(np.round(real_sol, 4))
        print("==== MD Risk Budgeting portfolio ====")
        print(np.round(self.y / sum(self.y), 4))
        print("==== MD Risk Budgeting portfolio wo ====")
        print(np.round(self.y, 4))
        if real_ES is not None: 
            print('==== Expected Shortfall of the portfolio ====')
            print(np.round(real_ES, 4))
        if real_VaR is not None: 
            print('==== VaR of the portfolio ====')
            print(np.round(real_VaR, 4))
        print('==== xi: VaR ====')
        print(round(self.xi/sum(self.y),4))
        print(round(self.c/sum(self.y),4))
                

    def weight_graph_y(self, real_weight= None):
        if real_weight is not None:
            for j in range(len(real_weight)):
                plt.axhline(real_weight[j], linestyle='--')
        for i in range(self.nb_asset):
            plt.plot(self.ys[i] / sum(self.y), label=f'Component {i+1}')
        plt.legend()
        plt.title('Evolution of Normalized Weights')
        plt.xlabel('Iterations')
        plt.ylabel('Weight Proportion')
        plt.show()


    def graph_y(self):
        for i in range(self.nb_asset):
                plt.plot(self.ys[i], label=f'Component {i+1}')
        plt.plot()
        
    def graph_cvar(self, real_ES = None, real_VaR = None):
        plt.plot(self.xi_final/ sum(self.y), label='VaR (xi)', color='red')
        plt.plot(self.c_final/ sum(self.y), label='CVaR (c)', color='blue')
        if real_ES is not None: 
                plt.axhline(real_ES, color='orange', linestyle='--')
        if real_VaR is not None: 
                plt.axhline(real_VaR, color='green', linestyle='--')    
                

    def convergence_graph(self):
        conv = self.differences * (np.arange(0,len(self.differences))**(1 - self.beta))
        plt.plot(conv)



class stoch_sol_measures:
    def __init__ (self, X, M, n_val, nb_asset, budgets,  n_epochs = 10, gamma=1, exp = 0.65, beta = 0.95, measure = 'VaR'):
        ones = np.ones(nb_asset)
        xi = 0
        k = 1
        ys = np.zeros((nb_asset, n_val*n_epochs))
        xi_final=[]
        y = budgets / np.std(X, axis=0)
        
        for j in tqdm(range(n_epochs)):
            np.random.shuffle(X)
            for i in range(n_val):    
                stepsize = (gamma /k** exp)
                y_min = min(min(y),1)
                if measure == 'VaR': 
                    Gamma_grad =  - (X[i] * ones * car_func(y,X[i],xi) / (ones - beta * ones)) - budgets/y 
                    xi = xi - stepsize*(1 - (car_func(y,X[i],xi) / (1 - beta)))
                    y = y*np.exp(-stepsize*y_min*Gamma_grad)
                elif measure == 'volatility':
                    func = - np.dot(y, X[i])- xi
                    Gamma_grad = -2 * ones * func * X[i] - budgets/y 
                    Gamma_xi = -2 * func 
                    xi = xi - stepsize * Gamma_xi
                    y = y*np.exp(-stepsize*y_min*Gamma_grad)
                elif measure == 'MAD':
                    indicator_pos = (-np.dot(y, X[i]) - xi >= 0)
                    indicator_neg = 1 - indicator_pos 
                    Gamma_grad = (-1 * indicator_pos + indicator_neg) * ones * X[i] - budgets/y 
                    Gamma_xi = -1*indicator_pos + indicator_neg
                    xi = xi - stepsize * Gamma_xi
                    y = y*np.exp(-stepsize*y_min*Gamma_grad)
                elif measure == 'variantile':
                    func = - np.dot(y, X[i]) - xi 
                    indicator_pos = (-np.dot(y, X[i]) - xi >= 0)
                    indicator_neg = 1 - indicator_pos 
                    Gamma_grad = - 2 * beta * indicator_pos * func * ones * X[i] - 2 * ones * (1-beta) * indicator_neg * func * X[i] -  budgets/y
                    Gamma_xi = -2 * func * beta * indicator_pos - 2 * (1-beta) * func * indicator_neg
                    xi = xi - stepsize * Gamma_xi
                    y = y*np.exp(-stepsize*y_min*Gamma_grad)
                else:
                    raise ValueError('The given risk measure is not applicable.')
                sum_y = np.sum(y)
                if sum_y>M:
                    y = M/sum_y*y
                for asset_index in range(nb_asset):
                    ys[asset_index, j*n_val + i] = y[asset_index] 
                xi_final.append(xi)
                k=k+1
        self.ys = ys
        self.xi = xi
        self.y = y
        self.xi_final = xi_final
        self.nb_asset = nb_asset
        self.beta = beta

    def results(self, real_sol=None, real_variance = None):
        if real_sol is not None:
            print('==== True Risk Budgeting portfolio ====')
            print(np.round(real_sol, 4))
        print("==== MD Risk Budgeting portfolio ====")
        print(np.round(self.y / sum(self.y), 4))
        print("==== MD Risk Budgeting portfolio wo ====")
        print(np.round(self.y, 4))
        if real_variance is not None: 
            print('==== volatilty of the portfolio ====')
            print(np.round(real_variance, 4))
        #print('==== volatility ====')
        #print(round(self.xi/sum(self.y),4)) #it is not really the volatilty, we need to introduce it in the function

                

    def weight_graph_y(self, real_weight= None):
        if real_weight is not None:
            for j in range(len(real_weight)):
                plt.axhline(real_weight[j], linestyle='--')
        for i in range(self.nb_asset):
            plt.plot(self.ys[i] / sum(self.y), label=f'Component {i+1}')
        plt.legend()
        plt.title('Evolution of Normalized Weights')
        plt.xlabel('Iterations')
        plt.ylabel('Weight Proportion')
        plt.show()


    def graph_y(self):
        for i in range(self.nb_asset):
                plt.plot(self.ys[i], label=f'Component {i+1}')
        plt.plot()
        
    def graph_cvar(self, real_ES = None, real_VaR = None):
        plt.plot(self.xi_final/ sum(self.y), label='VaR (xi)', color='red')
        plt.plot(self.c_final/ sum(self.y), label='CVaR (c)', color='blue')
        if real_ES is not None: 
                plt.axhline(real_ES, color='orange', linestyle='--')
        if real_VaR is not None: 
                plt.axhline(real_VaR, color='green', linestyle='--')    
                



class stoch_sol_highdim:
    def __init__ (self, X, M, n_val, nb_asset, budgets,  n_epochs = 10, gamma=1, exp = 0.65, beta = 0.95, func = False, gamma_star = None): 
        if func==True:
            if gamma_star is None:
                raise ValueError("gamma_star must be provided if func is True")
            self.gamma_star = gamma_star
        ones = np.ones(nb_asset)
        xi = 0
        k = 1
        c = 0
        y = budgets / np.std(X, axis=0)
        y_bar = np.zeros((1, nb_asset))
        xi_bar = 0
        
        for j in tqdm(range(n_epochs)):
            np.random.shuffle(X)
            for i in range(n_val):    
                stepsize = (gamma /k)** exp
                y_min = min(min(y),1)
                Gamma_grad =  - (X[i] * ones * car_func(y,X[i],xi) / (ones - beta * ones)) - budgets/y 
                c = c - 1/(k+1)*(c-(xi + 1/(1-beta)*pos_func(y,X[i],xi)))
                xi = xi - stepsize*(1 - (car_func(y,X[i],xi) / (1 - beta)))
                y = y*np.exp(-stepsize*y_min*Gamma_grad)
                sum_y = np.sum(y)
                if sum_y>M:
                    y = M/sum_y*y
                if func: 
                    y_bar= y_bar * i/(i+1) + y *  1/(i+1)
                    xi_bar = xi_bar * i/(i+1) + xi *  1/(i+1)    
                k=k+1
        self.xi = xi
        self.c = c
        self.y = y
        self.nb_asset = nb_asset
        self.beta = beta
        self.y_bar = y_bar
        self.xi_bar = xi_bar
        if func:
            self.func = np.mean(xi_bar + 1/(1-beta) * np.maximum(-np.dot(y_bar,X.T) - xi_bar,0) - np.sum(budgets* np.log(y_bar))) - self.gamma_star

    def results(self, real_sol=None, real_ES = None, real_VaR = None):
        if real_sol is not None:
            print('==== True Risk Budgeting portfolio ====')
            print(np.round(real_sol, 4))
        print("==== MD Risk Budgeting portfolio ====")
        print(np.round(self.y / sum(self.y), 4))
        print("==== MD Risk Budgeting portfolio wo ====")
        print(np.round(self.y, 4))
        if real_ES is not None: 
            print('==== Expected Shortfall of the portfolio ====')
            print(np.round(real_ES, 4))
        if real_VaR is not None: 
            print('==== VaR of the portfolio ====')
            print(np.round(real_VaR, 4))
        print('==== xi: VaR ====')
        print(round(self.xi/sum(self.y),4))
        print(round(self.c/sum(self.y),4))


class determ_sol:
    def __init__(self, SM_model, M, n_iter, nb_asset, budgets, gamma = 1, exp= 0.55,beta = 0.95, fixed_stepsize = False):
             
        np.random.seed(0)
        # choose risk budgets (equal)
        # initialize y
        y = budgets / ((np.diag(SM_model.scales_[0])* SM_model.dofs_[0])/ (SM_model.dofs_[0] -2)) 
        # store values
        f_values_list = [StudentMixtureExpectedShortfall(SM_model).expected_shortfall(y, beta) - budgets@np.log(y)]
        ys = np.zeros((nb_asset, n_iter))
        
        
        for i in tqdm(range(1, n_iter)):
            if fixed_stepsize:
                stepsize = 1
            else:
                stepsize= (gamma/i ** exp)
            Gamma_grad = StudentMixtureExpectedShortfall(SM_model).expected_shortfall_gradient(y, beta) - budgets/y
            y_min = min(1,min(y))
            y = y*np.exp(-stepsize*y_min*Gamma_grad)
            sum_y = sum(y) 
            if sum_y>M:
                y = M/sum_y*y
            f_value = StudentMixtureExpectedShortfall(SM_model).expected_shortfall(y, beta) - budgets@np.log(y)
            for asset_index in range(nb_asset):
                ys[asset_index, i] = y[asset_index] 
            f_values_list.append(f_value)

        self.f_values = f_values_list
        self.y = y
        self.ys = ys
        self.nb_asset = nb_asset
        self.beta = beta
        
#maybe a change has to be done as we don't take here the xbar
    def convergence(self, gamma_star):
        diff = np.array(self.f_values) - gamma_star
        conv = diff * (np.arange(1,diff.size+1)**(1-self.beta))
        plt.plot(conv)

    def results(self, real_sol=None): 
        if real_sol is not None:
            print('==== True Risk Budgeting portfolio ====')
            print(np.round(real_sol, 4))
        print("==== MD Risk Budgeting portfolio ====")
        print(np.round(self.y / sum(self.y), 4))

    def weight_graph_y(self, real_weight= None):
        if real_weight is not None:
            for j in range(len(real_weight)):
                plt.axhline(real_weight[j], linestyle='--')
        for i in range(self.nb_asset):
            plt.plot(self.ys[i] / sum(self.y), label=f'Component {i+1}')
        plt.legend()
        plt.title('Evolution of Normalized Weights')
        plt.xlabel('Iterations')
        plt.ylabel('Weight Proportion')
        plt.show()


    def graph_y(self):
        for i in range(self.nb_asset):
                plt.plot(self.ys[i], label=f'Component {i+1}')
        plt.plot()


class RiskBudgeting:
    """

    Representation of a Risk Budgeting problem.
    This class allows to find the Risk Budgeting portfolio for different risk measures under given additional
    specifications.

    Parameters
    ----------

    risk_measure : {'volatility' (default),
                    'median_absolute_deviation',
                    'expected_shortfall',
                    'power_spectral_risk_measure',
                    'variantile'}
        String describing the type of risk measure to use.

    budgets : {'ERC' (default), numpy.ndarray}
        String or array describing the risk budgets. 'ERC' stands for Equal Risk Contribution. In other cases, risk budgets
        should be given as an array with relevant dimension.

    expectation : bool, default to False.

    beta : float, defaults to 1.00
        Weight of the risk measure component when 'expectation' is True. Not used when 'expectation' is False.

    delta : float, defaults to 1.00
        Weight of the expected return component when 'expectation' is True. Not used when 'expectation' is False.

    alpha : float
        Confidence level when 'risk_measure' is 'expected_shortfall'. Weight of the first component when 'risk_measure'
        is 'variantile'. Not used in other cases.

    gamma : float
        Coefficient of the power utility function needed when 'risk_measure' is 'power_spectral_risk_measure' Not used
        in other cases.

    Attributes
    ----------
    x : numpy.ndarray
        The weights of the computed Risk Budgeting portfolio.

    ys: numpy.ndarray, default to None.
        If 'store' parameter in solve() function is True, store y vectors along the optimization path.

    ts: numpy.ndarray, default to None.
        If 'store' parameter in solve() function is True, store t values along the optimization path.
        
    """

    def __init__(self,
                 risk_measure='volatility',
                 budgets='ERC',
                 expectation=False,
                 beta=1.00,
                 delta=1.00,
                 alpha=None,
                 gamma=None
                 ):

        self.risk_measure = risk_measure
        self.budgets = budgets
        self.expectation = expectation
        self.beta = beta
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.ys = None
        self.ts = None
        self.success = None
        self.x = None

    def solve(self, X, epochs=None, minibatch_size=128, y_init=None, t_init=None, eta_0_y=None, eta_0_t=None, c=0.65,
              polyak_ruppert=0.2, discretize=None, proj_y=None, store=False):

        """

        Solves the defined risk budgeting problem using a given sample of asset returns via
        stochastic gradient descent and returns the risk budgeting portfolio.

        Parameters
        ----------

        X : numpy.ndarray shape=(n,d)
            Sample of asset returns

        epochs : int, optional. Defaults to int(2e06/n).
            Number of epochs.

        minibatch_size : int, optional. Defaults to 128.
            Mini-batch size.

        y_init (numpy.ndarray, optional): numpy.ndarray shape=(d,). Defaults to a vector which is a
            solution to risk budgeting problem for volatility under the assumption that the correlation matrix is
            all-ones matrix.
            Initial value for each element of the vector of asset weights.

        t_init : float, optional. Defaults to a minimizer of a similar problem with analytical solution.
            Initial value for t.

        eta_0_t : float, optional. Defaults to 0.5.
            Step size coefficient for variable t.

        eta_0_y : float, optional. Defaults to 50/d.
            Step size coefficient for vector y.

        c : float, optional. Defaults to 0.65.
            Step size power.

        polyak_ruppert : float, optional. Defaults to 0.2.
             Polyak-Ruppert averaging for last % iterates.

        discretize : dict, optional. Defaults to {'step': 50, 'bounds': (.5, .99)}
            Parameters to discretize the integral for spectral risk measures.

        proj_y : float, optional. Defaults to y_init.
            Value for projection of asset weights into the feasible space.

        store : bool, optional. Defaults to False.
            store y and t along the optimization path.

        """

        n, d = X.shape

        # Set budgets if ERC
        if type(self.budgets) == str:
            if self.budgets == 'ERC':
                self.budgets = np.ones(d) / d

        if False in self.budgets > 0 or True in self.budgets >= 0:
            raise ValueError('The budgets should be in the range (0,1).')

        # Choose number of epochs based on sample size
        if epochs is None:
            epochs = int(2e06 / n)

        # Initialize y
        if y_init is None:
            y = self.budgets / np.std(X, axis=0)
        else:
            y = y_init

        if proj_y is None:
            proj_y = y

        # Set step size coefficients for y and t
        if eta_0_y is None:
            eta_0_y = 50 / d
        if eta_0_t is None:
            eta_0_t = .5

        if self.beta <= 0:
            raise ValueError('beta should greater than 0.')

        # Needed for Polyak-Ruppert averaging
        y_sum = np.zeros(d)
        sum_k_first = int((1 - polyak_ruppert) * (epochs * n / minibatch_size))

        # Store along the optimization path
        y_ = [y]
        k = 0

        if self.risk_measure == 'volatility':
            # Initialize t
            if t_init is None:
                t = np.dot(np.dot(y, np.cov(X, rowvar=False)), y)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):

                    # Mini-batch
                    x = X[i:i + minibatch_size]

                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c

                    # Gradient
                    r = np.dot(y, x.T)
                    grad_t = np.mean(self.beta * -2 * (r - t))
                    grad_y = np.mean(self.beta *
                                     2 * (r - t).reshape((x.shape[0], 1)) * x - self.budgets / y -
                                     self.delta * self.expectation * x, axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        elif self.risk_measure == 'median_absolute_deviation':
            # Initialize t
            if t_init is None:
                t = np.dot(np.dot(y, np.cov(X, rowvar=False)), y)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):

                    # Mini-batch
                    x = X[i:i + minibatch_size]

                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c

                    # Gradient
                    indicator_pos = (np.dot(y, x.T) - t >= 0).reshape((x.shape[0], 1))
                    indicator_neg = 1 - indicator_pos
                    grad_t = np.mean(self.beta * -1 * indicator_pos + indicator_neg)
                    grad_y = np.mean(self.beta * x * (
                            indicator_pos - indicator_neg) - self.budgets / y - self.delta * self.expectation * x,
                                     axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        elif self.risk_measure == 'expected_shortfall':
            # Initialize t
            if t_init is None:
                t = -np.dot(y, np.mean(X, axis=0)) + np.dot(np.dot(y, np.cov(X, rowvar=False)), y) * norm.ppf(
                    self.alpha)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):

                    # Mini-batch
                    x = X[i:i + minibatch_size]

                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c

                    # Gradient
                    indicator = (-np.dot(y, x.T) - t >= 0).reshape((x.shape[0], 1))
                    grad_t = np.mean(self.beta * 1 - (1 / (1 - self.alpha)) * indicator)
                    grad_y = np.mean(self.beta *
                                     (-x / (
                                             1 - self.alpha)) * indicator - self.budgets / y + self.delta * self.expectation * x,
                                     axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        elif self.risk_measure == 'power_spectral_risk_measure':
            # Initialize t
            if discretize is None:
                discretize = {'step': 50, 'bounds': (.5, .99)}
            u = np.linspace(discretize['bounds'][0], discretize['bounds'][1], discretize['step'])
            w = (self.gamma * u ** (self.gamma - 1))  # power law
            delta_w = np.diff(w)
            u = u[1:]
            if t_init is None:
                t = -np.dot(y, np.mean(X, axis=0)) + np.dot(np.dot(y, np.cov(X, rowvar=False)), y) * norm.ppf(u)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):

                    # Mini-batch
                    x = X[i:i + minibatch_size]

                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c

                    # Gradient
                    indicator = (-np.dot(y, x.T)[:, None] - t >= 0)
                    grad_t = np.mean(self.beta * delta_w * (1 - u) - delta_w * indicator, axis=0)
                    grad_y = np.mean(self.beta *
                                     -np.dot(delta_w, indicator.T).reshape(
                                         (x.shape[0], 1)) * x - self.budgets / y + self.delta * self.expectation * x,
                                     axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        elif self.risk_measure == 'variantile':
            # Initialize t
            if t_init is None:
                t = -np.dot(np.dot(y, np.cov(X, rowvar=False)), y)
            else:
                t = t_init
            t_ = [t]
            for s in range(epochs):
                np.random.shuffle(X)
                for i in range(0, n, minibatch_size):
                    # Mini-batch
                    x = X[i:i + minibatch_size]
                    # Step size schedule
                    eta_t = eta_0_t / (1 + k) ** c
                    eta_y = eta_0_y / (1 + k) ** c
                    # Gradient
                    loss = -np.dot(y, x.T)
                    indicator_pos = (loss - t >= 0).reshape((x.shape[0], 1))
                    indicator_neg = (loss - t < 0).reshape((x.shape[0], 1))
                    grad_t = np.mean(
                        self.beta * -2 * self.alpha * (loss - t) * indicator_pos + -2 * (1 - self.alpha) * (
                                loss - t) * indicator_neg)
                    grad_y = np.mean(self.beta *
                                     -2 * self.alpha * (loss - t).reshape((x.shape[0], 1)) * x * indicator_pos + -2 * (
                                             1 - self.alpha) * (loss - t).reshape(
                        (x.shape[0], 1)) * x * indicator_neg - self.budgets / y, axis=0)

                    # Descent
                    t = t - eta_t * grad_t
                    y = y - eta_y * grad_y
                    y = np.where(y <= 0, proj_y, y)

                    if k + 1 > sum_k_first:
                        y_sum += y

                    if store:
                        y_.append(y)
                        t_.append(t)

                    k += 1

        else:
            raise ValueError('The given risk measure is not applicable.')

        y_sgd = y_sum / int(polyak_ruppert * (epochs * n / minibatch_size))
        theta_sgd = y_sgd / y_sgd.sum()

        self.x = theta_sgd

        if store:
            self.ys = y_
            self.ts = t_


class StudentMixtureExpectedShortfall:

    def __init__(self, model):
        self.model = model

    def value_at_risk(self, theta, alpha, method='Newton', a=-100, b=100, tol=1e-6, max_iter=1000):
        probs = self.model.weights_
        means = np.dot(self.model.locations_, theta)
        scales = np.sqrt(np.dot(np.dot(self.model.scales_, theta), theta))
        dofs = self.model.dofs_

        if method == 'Newton':
            var = 0.
            for _ in range(10):
                eq = np.dot(probs, t.cdf((var + means) / scales, df=dofs))
                diff = np.dot(probs / scales, t.pdf((var + means) / scales, df=dofs))
                var = var - (eq - alpha) / diff
            return var

        elif method == 'bisection':
            i = 0
            while (b - a) / 2 > tol and i < max_iter:
                f_a = np.dot(probs, t.cdf((a + means) / scales, df=dofs)) - alpha
                f_b = np.dot(probs, t.cdf((b + means) / scales, df=dofs)) - alpha

                c = (a + b) / 2
                f_c = np.dot(probs, t.cdf((c + means) / scales, df=dofs)) - alpha

                if np.sign(f_c) == np.sign(f_a):
                    a = c
                else:
                    b = c
                i += 1
            return c

    def expected_shortfall(self, theta, alpha):
        var = self.value_at_risk(theta, alpha)
        probs = self.model.weights_
        means = np.dot(self.model.locations_, theta)
        scales = np.sqrt(np.dot(np.dot(self.model.scales_, theta), theta))
        dofs = self.model.dofs_
        z = (var + means) / scales
        first_comp = scales * (dofs + z ** 2) / (dofs - 1.) * t.pdf(z, df=dofs)
        second_comp = means * t.cdf(-z, df=dofs)
        es = np.dot(probs, first_comp - second_comp) / (1 - alpha)
        return es

    def solve_risk_budgeting(self, budgets, alpha, on_simplex=False, kappa=1, method=None, maxiter=15000):
        func = lambda y: self.expected_shortfall(y, alpha) - kappa * budgets @ np.log(y)
        d = budgets.shape[0]
        bounds = [(1e-8, None) for _ in range(d)]
        if on_simplex:
            constraints = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1})
        else:
            constraints = None
        # solve
        optim_res = minimize(func, x0=budgets, bounds=bounds, constraints=constraints, method=method, options={'maxiter': maxiter})
        # normalize if necessary
        if on_simplex:
            port_rb = optim_res.x
        else:
            port_rb = optim_res.x/sum(optim_res.x)
        return port_rb, optim_res

    def expected_shortfall_gradient(self, theta, alpha):

        def h(weights, conf_int):
            return (self.value_at_risk(weights, conf_int) + np.dot(self.model.locations_, weights)) / np.sqrt(
                np.dot(np.dot(self.model.scales_, weights), weights))

        def value_at_risk_gradient(weights, conf_int):
            probs = self.model.weights_
            means = np.dot(self.model.locations_, weights)
            scales = np.sqrt(np.dot(np.dot(self.model.scales_, weights), weights))
            dofs = self.model.dofs_

            h_ = h(weights, conf_int)

            num = np.sum(probs * t.pdf(h_, df=dofs) * (
                    (np.dot(self.model.scales_, weights).T * h_) / scales ** 2 - self.model.locations_.T / scales),
                         axis=1)
            denum = np.sum(probs * t.pdf(h_, df=dofs) / scales)

            return num / denum

        def h_gradient(theta, alpha):
            probs = self.model.weights_
            means = np.dot(self.model.locations_, theta)
            scales = np.sqrt(np.dot(np.dot(self.model.scales_, theta), theta))
            dofs = self.model.dofs_

            var_grad = value_at_risk_gradient(theta, alpha)
            var_ = self.value_at_risk(theta, alpha)

            return ((self.model.locations_ + var_grad).T * scales - np.dot(self.model.scales_, theta).T * (
                    var_ + means) / scales) / scales ** 2

        def t_density_gradient(x):
            dofs = self.model.dofs_
            return -(dofs + 1) * t.pdf(x, df=dofs) * (x / (dofs + x ** 2))

        probs = self.model.weights_
        means = np.dot(self.model.locations_, theta)
        scales = np.sqrt(np.dot(np.dot(self.model.scales_, theta), theta))
        dofs = self.model.dofs_

        h_ = h(theta, alpha)
        h_grad = h_gradient(theta, alpha)

        return np.sum(probs / (1 - alpha) * ((np.dot(self.model.scales_, theta).T / (scales * (dofs - 1)) * (
                dofs + h_ ** 2) + 2 * h_ * h_grad * scales / (dofs - 1)) * t.pdf(h_, df=dofs) + (
                                                     (dofs + h_ ** 2) * (
                                                     scales / (dofs - 1))) * t_density_gradient(h_) * h_grad - (
                                                     self.model.locations_.T * t.cdf(-1 * h_,
                                                                                     df=dofs) - means * t.pdf(
                                                 -1 * h_, df=dofs) * h_grad)), axis=1)


class AlternativeMethods:

    def __init__(self):
        self.x = None
        self.success = None
        self.ys = None

    def osbgd(self, sample, budgets, alpha, epsilon=1e-04, tol=1e-05, max_iter=1000,
              store=False, printer=False):
        nb_asset = len(budgets)

        # Create and initialize variables
        tols = []
        y_ = []
        fvalue = []

        # Initialize y
        y = proj_y = budgets / np.std(sample, axis=0)

        # Create matrix for efficient calculation of ES gradient by Finite-Difference
        eps_diag = np.identity(nb_asset) * epsilon
        eps_diag = np.concatenate([np.zeros((nb_asset, 1)), eps_diag], axis=1)

        i = 0
        dist = np.inf
        func = 0

        while dist > tol or i < 5:

            # Monte Carlo ES and ES sensitivity using Finite-Difference
            es_ = sample_expected_shortfall(np.dot(sample, eps_diag + y.reshape(nb_asset, 1)), alpha)
            es = es_[0]
            es_eps = es_[1:]
            grad_es = (es_eps - es) / epsilon

            # grad f
            func_prev = func
            func = es - budgets @ np.log(y)
            grad = grad_es - budgets / y

            # Barzilai-Borwein step size
            if i > 0:
                y_k_1 = y_k
                y_k = y

                grad_k_1 = grad_k
                grad_k = grad

                eta = np.linalg.norm(y_k - y_k_1, ord=2) ** 2 / np.abs(np.dot(y_k - y_k_1, grad_k - grad_k_1))
            else:
                eta = 1
                y_k = y
                grad_k = grad

            # Descent
            y = y - eta * grad
            y = np.where(y <= 0, proj_y, y)

            i += 1

            # Stopping rule
            dist = np.abs(func - func_prev)
            # dist = np.linalg.norm(grad, ord=np.inf)
            if store:
                tols.append(dist)
                y_.append(y)
                fvalue.append(func)
            if printer:
                print('func: ', func, 'norm grad: ', dist, ', eta: ', eta)
            if i > max_iter:
                success = False
                print('Number of maximum iterations is reached: ', max_iter, ' iterations')
                return y / y.sum(), y_, fvalue, tols, success
        success = True
        self.x = y / y.sum()
        self.success = success
        self.ys = y_

    def msbgd(self, model, budgets, alpha, n_sample, epsilon=1e-04, n_avg=10, n_iter=50, store=False, printer=False):

        nb_asset = len(budgets)

        # Create and initialize variables
        y_ = []
        fvalue = []

        # Initialize y
        sample = model.rvs(n_sample)
        y = proj_y = budgets / np.std(sample, axis=0)

        # Create matrix for efficient calculation of ES gradient by Finite-Difference
        eps_diag = np.identity(nb_asset) * epsilon
        eps_diag = np.concatenate([np.zeros((nb_asset, 1)), eps_diag], axis=1)

        for i in range(n_iter):

            sample = model.rvs(n_sample)

            # Monte Carlo ES and ES sensitivity using Finite-Difference
            es_ = sample_expected_shortfall(np.dot(sample, eps_diag + y.reshape(nb_asset, 1)), alpha)
            es = es_[0]
            es_eps = es_[1:]
            grad_es = (es_eps - es) / epsilon

            # grad f
            func = es - budgets @ np.log(y)
            grad = grad_es - budgets / y

            # Barzilai-Borwein step size
            if i > 0:
                y_k_1 = y_k
                y_k = y

                grad_k_1 = grad_k
                grad_k = grad

                eta = np.linalg.norm(y_k - y_k_1, ord=2) ** 2 / np.abs(np.dot(y_k - y_k_1, grad_k - grad_k_1))
            else:
                eta = 1
                y_k = y
                grad_k = grad

            # Descent
            y = y - eta * grad
            y = np.where(y <= 0, proj_y, y)

            if store:
                y_.append(y)
                fvalue.append(func)
            if printer:
                print('func: ', func, ', eta: ', eta)

        theta = np.mean(y_[-n_avg:], axis=0)
        theta = theta / theta.sum()

        self.x = theta
        self.ys = y_
