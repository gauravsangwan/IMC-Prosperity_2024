## ROUND 1 LOGS

# STARFRUIT

## Possible Idea

For Starfruit since, having a positin limit of **20**, and since the price oscillates a lot with the range depicted in Season2 Episode 2 being 4176 to 6423. 
I will try to run a linear regression on the final few timesteps of starfruit prices to predict the next price. 



---
# AMETHYSTS

## Possible Ideas

Aquatic Amethysts, have a position limit of **20**, and it is <ins>hypothesized</ins> that the price doesnot oscillates must and we must play on bid ask spread. 


---
# Manual trading - SCUBA GEAR

- There are two rounds of bidding.
- Each goldfish has a reserve price for their scuba gear, which is uniformly distributed between 900 and 1000 seashells.
- The goldfish will accept the lowest bid that is over their reserve price.
- The goal is to make offers that maximize the chances of acquiring the scuba gear at a price that's acceptable to the goldfish but still profitable for the buyer.

The Probability Distribution function (PDF(x)) as described above is : 
$$ PDF(x) = x/100 - 9 , 900 <= x <= 1000 and 0 otherwise

Therefore, the CDF will be 

$$ CDF(x) = x^2/200 - 9x + 4050 900<=x<=1000 and 0 if x<900 and 950 if x>1000


As shown in scuba.ipynb we simulated 10 million iterations of a monte-carlo simulation, and we got the mean as 969, standard deviation as 23.68 and median as 971. 

Using **Mean Based Strategy**, One of my big will be around mean 970 and the other will be mean - std_dev, which is 946.
