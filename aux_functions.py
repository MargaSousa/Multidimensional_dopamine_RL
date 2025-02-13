import numpy as np
import pdb
from scipy.optimize import curve_fit
import scipy
from sklearn.linear_model import LinearRegression
import pandas as pd

f_reward = lambda x, a: a*x

# Estimate reversal point and posivite and negative slope
# Edited from code provided in Dabney et al 2020
def get_estimated_expectile(reward_amounts_control,responses):

    k=np.linspace(-0.5,reward_amounts_control.shape[0]-0.5,reward_amounts_control.shape[0]+1)
    critvals = np.zeros(k.shape[0])


    responses_all_amounts = np.array([])
    amounts = np.array([])

    # Extreme values
    critvals[0]=np.sum(responses>0)


    for i,r in enumerate(reward_amounts_control):
        critvals[i+1]=np.sum(responses[(i+1):,:]>0)+np.sum(responses[:i+1,:]<0)

        res=responses[i,:]
        res=res[~np.isnan(res)]

        responses_all_amounts = np.concatenate((responses_all_amounts, res))
        amounts = np.concatenate((amounts, np.ones(len(res)) * reward_amounts_control[i]))


    res_last=responses[:,-1]
    res_last=res_last[~np.isnan(res_last)]
    responses_all_amounts=np.concatenate((responses_all_amounts,res_last))
    amounts = np.concatenate((amounts, np.ones(len(res_last)) * reward_amounts_control[-1]))


    critvals=critvals+0.001*np.random.rand(critvals.shape[0])
    mcv=np.argmax(critvals)
    zc=k[mcv]

    if zc<0:
        reversal_point=reward_amounts_control[0]+0.1 # arbitrary, we don't know
        tau=0
        x_pos = amounts[amounts > reversal_point]-reversal_point
        y_pos = responses_all_amounts[amounts > reversal_point]
        x_pos=x_pos.astype(float)
        y_pos=y_pos.astype(float)
        popt_pos, pcov_pos = curve_fit(f_reward, x_pos, y_pos,bounds=[0, np.inf])
        popt_pos=popt_pos[0]
        pcov_pos=pcov_pos[0][0]
        popt_neg=np.nan
        con=popt_pos

    elif zc>reward_amounts_control.shape[0]-1:
        reversal_point=reward_amounts_control[-1]-0.1 # arbitrary, we don't know
        tau=1
        x_neg = amounts[amounts < reversal_point]-reversal_point
        x_neg=x_neg.astype(float)
        y_neg = responses_all_amounts[amounts < reversal_point]
        y_neg=y_neg.astype(float)
        popt_neg, pcov_neg = curve_fit(f_reward, x_neg, y_neg,bounds=[0, np.inf])
        popt_neg=popt_neg[0]
        pcov_neg=pcov_neg[0][0]
        popt_pos=np.nan
        con=popt_neg/tau

    else:
        neighbors=np.array([critvals[mcv-1],critvals[mcv],critvals[mcv+1]])
        w=np.abs(np.diff(neighbors))
        w=(1./w)/np.sum((1./w))
        reversal_point=w[0]*reward_amounts_control[int(zc-0.5)]+w[1]*reward_amounts_control[int(zc+0.5)]

        # Get slopes
        x_pos = amounts[amounts > reversal_point]-reversal_point
        x_pos=x_pos.astype(float)
        x_neg = amounts[amounts < reversal_point]-reversal_point
        x_neg=x_neg.astype(float)
        y_pos = responses_all_amounts[amounts > reversal_point]
        y_pos=y_pos.astype(float)
        y_neg = responses_all_amounts[amounts < reversal_point]
        y_neg=y_neg.astype(float)


        # Set the reversal point
        reg_neg=LinearRegression(fit_intercept=False,positive=True).fit(x_neg.reshape(-1,1),y_neg)
        reg_pos=LinearRegression(fit_intercept=False,positive=True).fit(x_pos.reshape(-1,1),y_pos)
        popt_neg=reg_neg.coef_[0]
        popt_pos=reg_pos.coef_[0]
        tau = np.abs(popt_pos) / (np.abs(popt_pos) + np.abs(popt_neg))
        con=popt_pos/tau
    if reversal_point<1:
        popt_neg=np.nan
    return reversal_point,tau,popt_neg,popt_pos,con


# Parametrized discount function
def exponential(x, b, s):
    return b*np.exp(-s * x)


# From Dabney et al 2020
def loss_fn(expectiles, taus,fr_neurons,constant,samples):
    """Sum of error in estimation. """

    delta = (samples[None, :] - expectiles[:, None])
    indic = np.array(delta <= 0., dtype=np.float32)
    grad = np.abs(taus[:, None] - indic) * delta
    grad=grad*constant[:,None]

    return np.mean(np.square(fr_neurons-grad)*w)


# From Dabney et al 2020
def get_dist(reversal_points, taus,fr_neurons,constant, minv=0., maxv=1., method=None,
                 max_samples=1000, max_epochs=10, N=100):
    """Decode reward amount given reversal points, asymmetries and firing rates."""

    ind = list(np.argsort(reversal_points))
    points = reversal_points[ind]
    tau = taus[ind]


    # Robustified optimization to infer distribution
    # Generate max_epochs sets of samples,
    # each starting the optimization at the best of max_samples initial points.
    sampled_dist = []
    for _ in range(max_epochs):
        # Randomly search for good initial conditions
        # This significantly improves the minima found
        samples = np.random.uniform(minv, maxv, size=(max_samples, N))
        fvalues = np.array([loss_fn(points, tau, fr_neurons,constant, x0) for x0 in samples])
        # Perform loss minimizing on expectile loss (w.r.t samples)
        x0 = np.array(sorted(samples[fvalues.argmin()]))
        fn_to_minimize = lambda x: loss_fn(points,tau,fr_neurons,constant,x)
        result = scipy.optimize.minimize(
            fn_to_minimize, method=method,
            bounds=[(minv, maxv) for _ in x0], x0=x0)['x']

        sampled_dist.extend(result.tolist())

    return sampled_dist, loss_fn(points,tau,w,fr_neurons,constant, np.array(sampled_dist))


# From Dabney et al 2020
def expectile_loss_fn(expectiles, taus,w,samples):
  """Expectile loss function, corresponds to distributional TD model """
  # distributional TD model: delta_t = (r + \gamma V*) - V_i
  # expectile loss: delta = sample - expectile
  delta = (samples[None, :] - expectiles[:, None])

  # distributional TD model: alpha^+ delta if delta > 0, alpha^- delta otherwise
  # expectile loss: |taus - I_{delta <= 0}| * delta^2

  # Note: When used to decode we take the gradient of this loss,
  # and then evaluate the mean-squared gradient. That is because *samples* must
  # trade-off errors with all expectiles to zero out the gradient of the
  # expectile loss.
  indic = np.array(delta <= 0., dtype=np.float32)
  grad = -0.5 * np.abs(taus[:, None] - indic) * delta
  return np.mean(np.square(np.mean(grad, axis=-1))*w)

# From Dabney et al 2020
def run_decoding_magnitude(reversal_points, taus,w, minv=0., maxv=1., method=None,
                 max_samples=1000, max_epochs=10, N=100):
  """Run decoding given reversal points and asymmetries (taus)."""

  ind = list(np.argsort(reversal_points))
  points = reversal_points[ind]
  tau = taus[ind]
  w=w[ind]

  # Robustified optimization to infer distribution
  # Generate max_epochs sets of samples,
  # each starting the optimization at the best of max_samples initial points.
  sampled_dist = []
  for _ in range(max_epochs):
    # Randomly search for good initial conditions
    # This significantly improves the minima found
    samples = np.random.uniform(minv, maxv, size=(max_samples, N))
    fvalues = np.array([expectile_loss_fn(points,tau,w,x0) for x0 in samples])

    # Perform loss minimizing on expectile loss (w.r.t samples)
    x0 = np.array(sorted(samples[fvalues.argmin()]))
    fn_to_minimize = lambda x: expectile_loss_fn(points,tau,w,x)
    result = scipy.optimize.minimize(
        fn_to_minimize, method=method,
        bounds=[(minv, maxv) for _ in x0], x0=x0)['x']

    sampled_dist.extend(result.tolist())

  return sampled_dist, expectile_loss_fn(points, tau,w,np.array(sampled_dist))


def run_decoding_time(time, discount, variance, population_responses,alpha):
    """Run decoding in time given temporal discounts, population responses where each neuron's
    response has a certain variance and a smoothing parameter alpha."""
    n_neurons=len(discount)
    n_time=len(time)
    inverse_variance = 1.0 / variance
    F = np.zeros((n_neurons, n_time))
    for i_t, t in enumerate(time):
        F[:, i_t] = (discount ** t)
    F = F * inverse_variance[:, None]
    U, s, vh = np.linalg.svd(F, full_matrices=True)
    L = np.min([vh.shape[0], np.shape(U)[1]])

    prob_time = np.zeros(n_time)
    for l in range(L):
        prob_time += (s[l] ** 2 / (s[l] ** 2 + alpha ** 2)) * np.dot(U[:, l],population_responses * inverse_variance) * vh[l,:] / s[l]

    prob_time[prob_time < 0] = 0
    prob_time = prob_time / np.sum(prob_time)

    return prob_time

def quadratic_function(x,a,b,c):
    return a+b*(x-c)**2


def get_expectiles(x,probability,taus):
    """Compute expectiles with levels taus from probability distribution defined over x."""
    expectiles = []
    for i_x, x_value in enumerate(x):
        expectation_left = np.nansum((x_value - x[:i_x]) * probability[:i_x])
        expectation_right = np.nansum((x[i_x + 1:] - x_value) * probability[i_x + 1:])
        level = expectation_left / (expectation_right + expectation_left)
        expectiles.append(level)
    expectiles = np.array(expectiles)
    chosen_expectiles = []
    pos_expectiles=[]
    for q in taus:
        pos = np.where(expectiles >= q)[0][0]
        e = x[pos]
        chosen_expectiles.append(e)
        pos_expectiles.append(pos)
    chosen_expectiles = np.array(chosen_expectiles)
    return pos_expectiles,chosen_expectiles

# To parse csv list properly
def safe_literal_eval(val):
    if pd.isna(val):  # Check if the value is NaN
        return None
    try:
        # Parse the list string using ast.literal_eval
        parsed_list = eval(val)
        # Replace any occurrence of 'nan' (or None after replacement) with np.nan
        parsed_list = [np.nan if x == 'nan' or x is None else x for x in parsed_list]
        return parsed_list
    except (ValueError, SyntaxError):
        return val  # Return the original value if it can't be parsed


def moving_average(y, mean_kernel, bin_width, kernel_type):
    """Outputs the moving average of a signal y with a kernel of kernel type with mean
    mean_kernel and bin width bin_width."""
    n_bins = (mean_kernel / bin_width) * 9
    bins = np.arange(n_bins / 2)
    x = bins * bin_width

    if kernel_type == "boxcart":
        kernel = x * 0
        kernel[x < mean_kernel] = 1
    else:
        decay = 1 / mean_kernel
        kernel = np.exp(-x * decay)

    kernel = kernel / np.sum(kernel)
    kernel = np.concatenate((np.zeros(int(n_bins / 2) - 1), kernel))
    return np.convolve(y, kernel, 'same')





