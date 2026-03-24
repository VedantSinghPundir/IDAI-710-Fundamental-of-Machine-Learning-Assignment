def mean(np_array):
    '''Your code here'''
    n = len(np_array)
    total = sum(np_array)
    ans = total / n
    '''Stop coding here'''    
    return ans


def stdev(np_array, mu='none'):
    '''Your code here'''
    n = len(np_array)
    total = sum(np_array)
    mean = total/n
    varience = 0
    for x in np_array:
        varience = (x - mean)**2 + varience
    varience = varience / n
    import math
    ans = math.sqrt(varience)
    '''Stop coding here'''    
    return ans

def sampleMean(np_array):
    ''' Each column represents a feature'''
    '''Your code here'''
    number_samples = len(np_array)
    number_features = len(np_array[0])
    ans = []

    for i in range(number_features):
        total = 0
        for j in range(number_samples):
            total = np_array[j][i] + total
        mean = total/number_samples
        ans.append(mean)
    '''Stop coding here'''    
    return ans


def covariance(np_array):
    ''' Each column represents a feature'''
    '''Your code here'''
    number_samples = len(np_array)
    number_features = len(np_array[0])
    means = sampleMean(np_array)
    
    ans = []
    for i in range(number_features):
        row = []
        for j in range(number_features):
            row.append(0)
        ans.append(row)
    
    for i in range(number_features):
        for j in range(number_features):
            total = 0
            for k in range(number_samples):
                total = total + (np_array[k][i] - means[i]) * (np_array[k][j] - means[j])
            ans[i][j] = total / number_samples
    '''Stop coding here'''    
    return ans