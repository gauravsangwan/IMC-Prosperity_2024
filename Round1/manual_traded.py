import random

# Define selling price and minimum profit
selling_price = 1000
min_profit = 100


minimum_bid = 900
maximum_bid = 1001
# Define offer prices (considering minimum profit)
offer_prices = []
for i in range(minimum_bid, maximum_bid+1):
  offer_prices.append(i)
# Simulate multiple goldfish with random reserve prices between 900 and 1000 (inclusive)
num_goldfish = 1000
reserve_prices = random.choices(range(900, 1001), k=num_goldfish)
# print(reserve_prices)

# Function to calculate acceptance probability for each offer price
def calculate_acceptance_probability(offer_price, reserve_prices):
  accepted_count = 0
  for reserve_price in reserve_prices:
    if offer_price >= reserve_price:
      accepted_count += 1
  return accepted_count / num_goldfish  # Probability of acceptance

# Calculate acceptance probabilities for both offer prices
acceptance_probabilities = [calculate_acceptance_probability(price, reserve_prices.copy()) for price in offer_prices]

# Print results (assuming offer prices are offered sequentially)
# print("Offer Prices (SeaShells):", offer_prices)
print("Acceptance Probabilities:", acceptance_probabilities)

# Note: This is a simulation and doesn't account for individual negotiation strategies.
