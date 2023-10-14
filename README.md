<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
        }
    });
    </script>
      
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

***

# Maximum Likelihood Estimation

**A powerfull parametric method**

*Posted on August 2023*

***

There are many ways to estimate parameters of a statistical model. One of which is maximum likelihood estimation. This method led to a powerful estimation not only for a normally distributed process but also other kinds of distribution. It estimates parameters by the use of likelihood function. What is the likelihood function?

## Likelihood Estimation

The process that we are trying to model can be described by

$$p(y\mid\theta{}) = Y$$

This model read as: given a parameter $\theta{}$, the probability that this process result in $Y$ is $p(y\mid\theta{})$. But we do not know the real parameters. We have our uncertainty about our guesses of the true parameters. So we estimate the parameter by looking at the data or observing the joint pdf $p(y\mid\theta{})$. Or in another word, we make $y$ fixed for a certain range

$$L(\theta{}\mid y)$$

This is what is called likelihood function. It tells us the likelihood of parameter theta given the data that we observed. And its value is the same as the joint pdf $p(y\mid\theta{})$ for all available $\theta{}$.

It is a distribution because it is a random variable. But we can select theta that has the highest likelihood of producing the data by finding the highest peak of this distribution. This is what is called maximum likelihood estimation.

## Maximum Likelihood Estimation

If the likelihood function is continuous and differentiable, then we can use calculus to find $\theta{}$ that has the highest likelihood. But in practice, numerical methods are used by our computer to iteratively try different sets of theta. Below is an example of the use of maximum likelihood estimation to model California house value in python


```python
import pandas as pd
from scipy.stats import gamma
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv("/content/sample_data/california_housing_train.csv")
```

Let's use histogram to plot the median house value so we have an idea about the distribution of the data.


```python
sns.set(rc={'figure.figsize':(12,8)})
sns.displot(data=df, x="median_house_value", binwidth=10000)
plt.xlim(0, 600000)
plt.show()
```


    
![png](output_14_0.png)
    


We see that this process, California house value might not have a normally distributed distribution. Further, this notebook will use gamma function to model this process.

Also, we see that there are outliers with house value above 500000. Before we go further we need to remove this outliers so that our articial model best expressed the process. We will just use a simple outliers remove which remove all value above 500000.


```python
df = df[df["median_house_value"] < 500000]
```

After we removed the outliers, the histogram now looks like this.


```python
sns.set(rc={'figure.figsize':(12,8)})
sns.displot(data=df, x="median_house_value", binwidth=10000)
plt.xlim(0, 600000)
plt.show()
```


    
![png](output_19_0.png)
    


Scipy already has numbers of amazing function that we can use to many distributions. To do maximum likelihood estimation in scipy, we can just call fit on the function that we assumed the distribution of the process.


```python
a, loc, scale = gamma.fit(df["median_house_value"])
print("a: {}".format(a))
print("Location: {}".format(loc))
print("Scale: {}".format(scale))
```

    a: 3.280774292048559
    Location: 13028.695548223923
    Scale: 54601.07627116213
    

This is the gamma distribution parameters that the maximum likelihood estimation think that best expressed our data. Now how about we try to simulate new data based on this distribution?


```python
data_from_artificial_model = gamma.rvs(a=a, loc=loc, scale=scale, size=1000, random_state=123)
sns.set(rc={'figure.figsize':(12,8)})
sns.displot(data=data_from_artificial_model, binwidth=10000)
plt.xlim(0, 600000)
plt.show()
```


    
![png](output_23_0.png)
    


We can see, visually, that our articial model has more or less produce the same data as the real process.

And we want to estimate the range where 90% of the data fall we get


```python
low_ci, high_ci = gamma.interval(confidence=0.90, a=a, loc=loc, scale=scale)
print(f"Lower bound confidence interval: {low_ci}, Upper bound confidenec interval: {high_ci}")
```

    Lower bound confidence interval: 65706.8610206829, Upper bound confidenec interval: 379539.5286284525
    

## Why Maximum Likelihood Estimation?

Now the question is why do we even use maximum likelihood estimation as opposed to the plug-in estimate? First of all, this method can model the uncertainty of our estimate. Through the use of the likelihood functions we indirectly model our parameter within a certain range of value. The second reason is that this method is not only able to model normally distributed processes, but also other processes that have weird distributions. For example a logistic function. We can see how much logistic function is being used in modern statistics and machine learning.
