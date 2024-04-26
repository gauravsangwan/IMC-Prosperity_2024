# Summary / Important Points

- Main character is cocatoo, who brings latest headlines everyday and **they may contain easter eggs for contest**. 

## Game Mechanics 

- 15 days total length is divided into 5 rounds ..... each round lasts 72 hours. 
- All algorithms are independent among the playes. 
- 72 hours to submit the python program 
- Round timings are in UTC

| Event | Dates |
|---|---|
| Tutorial | Feb 12, 9:00 → Apr 8, 9:00 |
| Round 1 | Apr 8, 9:00 → Apr 11, 9:00 |
| Round 2 | Apr 11, 9:00 → Apr 14, 9:00 |
| Round 3 | Apr 14, 9:00 → Apr 17, 9:00 |
| Round 4 | Apr 17, 9:00 → Apr 20, 9:00 |
| Round 5 | Apr 20, 9:00 → Apr 23, 9:00 |

---

# Trading glossary

- Exchange : where buyers and sellers meet
- order : Owner + Product + quantity + price + validity
            for a order to execute conditions of the sell as well as the buy orders must meeet.

- bid order 
- ask order/offer
- order matching 
- order book 
- priority
- market making 


# Python Program tutorial

- algorithm class will be modified only. 
- **AIM: EARN AS MANY SEA SHELLS AS POSSIBLE **
- In ```Trader``` class we have a single method ```run``` which contains the **trading logic**. 

## TRADER CLASS

```
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData
```





## Links

- [STANFORD_CARDINAL](https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal)
- [Arima](https://github.com/kzqiu/imc-2023/tree/master)
- [Boolinger](https://github.com/Kratos-is-here/IMC-Trading-Prosperity)
- [Algo_visualiser](https://github.com/jmerle/imc-prosperity-2-visualizer)
- [Ranked57th](https://github.com/MichalOkon/imc_prosperity)
- [TEETAJP](https://github.com/teetajp/IMC-Prosperity)


https://www.youtube.com/watch?v=JN2LZ9YcT6k

https://www.youtube.com/watch?v=c0-I4ngbH1c