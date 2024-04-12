import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import collections 
from collections import defaultdict
import random
import math
import copy
import numpy as np

MAX_INT = int(1e9)

empty_dict = {'AMETHYSTS':0,'STARFRUIT':0}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

def default_value():
    return copy.deepcopy(empty_dict)

class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(default_value)
    person_actvalof_position = defaultdict(default_value)

    cpnl = defaultdict(lambda: 0)

    startfruit_cache = []
    startfruit_dim = 4
    steps = 0



    def calc_next_price_starfruit(self):
        #OLD
        coef = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892]
        intercept = 4.481696494462085
        #4D
        #coef = [-0.6563098,  -0.41592724, -0.24971836, -0.1220081 ]
        #intercept = 1.2886011626592033e-07
        #7D
#         coef = [-0.67067438, -0.44602041, -0.29958834, -0.20004169, -0.11883134, -0.05827812,
#  -0.01558487]
#         intercept = 1.4231017349099265e-07
        # Day Extended
        # coef = [-0.67657743, -0.437124,   -0.27277644, -0.1224234 ]
        # intercept = 5.460255106903702e-07
        
        nxt_price = intercept
        for i, val in enumerate(self.startfruit_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))
    
    def extract_values(self,order_dict,buy = 0):
        total_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            total_vol += vol
            if total_vol > mxvol:
                mxvol = vol
                best_val = ask
        return total_vol, best_val
    

    def compute_orders_amethyst(self,product,order_depth,acc_bid,acc_ask):
        orders: list[Order] = []
        
        order_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        order_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(),reverse=True))

        sell_vol, best_sell_price = self.extract_values(order_sell)
        buy_vol, best_buy_price = self.extract_values(order_buy,1)

        curr_pos = self.position[product]

        mx_with_buy = -1

        for ask,vol in order_sell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and curr_pos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy,ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - curr_pos)
                curr_pos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))
        
        mprice_actual = (best_sell_price + best_buy_price)/2
        mprice_ours = (acc_bid + acc_ask)/2

        undercut_buy = best_buy_price + 1
        undercut_sell = best_sell_price - 1

        bid_price = min(undercut_buy, acc_bid -1)
        sell_price = max(undercut_sell, acc_ask + 1)

        max_pos = 40

        if curr_pos < self.POSITION_LIMIT['AMETHYSTS'] and self.position[product] < 0:
            num = min(max_pos, self.POSITION_LIMIT['AMETHYSTS'] - curr_pos)
            orders.append(Order(product, min(undercut_buy+1,acc_bid-1),num))
            curr_pos += num
        
        if curr_pos < self.POSITION_LIMIT['AMETHYSTS'] and self.position[product] >15:
            num = min(max_pos, self.POSITION_LIMIT['AMETHYSTS'] - curr_pos)
            orders.append(Order(product, min(undercut_buy-1,acc_bid-1),num))
            curr_pos += num
        
        if curr_pos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(max_pos, self.POSITION_LIMIT['AMETHYSTS'] - curr_pos)
            orders.append(Order(product, bid_price,num))
            curr_pos += num
        
        curr_pos = self.position[product]

        for bid, vol in order_buy.items():
            if (bid > acc_ask or ((self.position[product]>0) and (bid == acc_ask))) and curr_pos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS'] - curr_pos)
                curr_pos += order_for
                assert(order_for <= 0) #Negative as we sell.
                orders.append(Order(product, bid, order_for))
        
        if (curr_pos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-max_pos, -self.POSITION_LIMIT['AMETHYSTS']-curr_pos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            curr_pos += num

        if (curr_pos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-max_pos, -self.POSITION_LIMIT['AMETHYSTS']-curr_pos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            curr_pos += num

        if curr_pos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-max_pos, -self.POSITION_LIMIT['AMETHYSTS']-curr_pos)
            orders.append(Order(product, sell_price, num))
            curr_pos += num

        return orders
    
    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.extract_values(osell)
        buy_vol, best_buy_pr = self.extract_values(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    def compute_orders(self, product, order_depth, acc_bid, acc_ask):

        if product == "AMETHYSTS":
            return self.compute_orders_amethyst(product, order_depth, acc_bid, acc_ask)
        
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {'AMETHYSTS': [], 'STARFRUIT': []}
        for key, val in state.position.items():
            self.position[key] = val
        for key, val in self.position.items():
            logger.print(f'{key} position: {val}')
        
        timestamp = state.timestamp

        if len(self.startfruit_cache) == self.startfruit_dim:
            self.startfruit_cache.pop(0)

        _, bs_starfruit = self.extract_values(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_starfruit = self.extract_values(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        self.startfruit_cache.append((bs_starfruit + bb_starfruit)/2)

        starfruit_lb = -MAX_INT
        starfruit_ub = MAX_INT

        if len(self.startfruit_cache) == self.startfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit() - 1
            starfruit_ub = self.calc_next_price_starfruit() + 1
        
        amethyst_lb = 10000
        amethyst_ub = 10000

        acc_bid = {'AMETHYSTS': amethyst_lb, 'STARFRUIT': starfruit_lb}
        acc_ask = {'AMETHYSTS': amethyst_ub, 'STARFRUIT': starfruit_ub}

        self.steps +=1 

        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                self.person_position[trade.buyer][product] =1.5
                self.person_position[trade.seller][product] = -1.5
                self.person_actvalof_position[trade.buyer][product] += trade.quantity
                self.person_actvalof_position[trade.seller][product] += -trade.quantity
        
        for product in ['AMETHYSTS','STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product,order_depth,acc_bid[product],acc_ask[product])
            result[product] += orders

        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp-100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.volume_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.cpnl[product] -= trade.quantity * trade.price
                else:
                    self.cpnl[product] += trade.quantity * trade.price

        totpnl = 0

        for product in state.order_depths.keys():
            settled_pnl = 0
            best_sell = min(state.order_depths[product].sell_orders.keys())
            best_buy = max(state.order_depths[product].buy_orders.keys())

            if self.position[product] < 0:
                settled_pnl += self.position[product] * best_buy
            else:
                settled_pnl += self.position[product] * best_sell
            totpnl += settled_pnl + self.cpnl[product]
            logger.print(f"For product {product}, {settled_pnl + self.cpnl[product]}, {(settled_pnl+self.cpnl[product])/(self.volume_traded[product]+1e-20)}")     

        
        logger.print(f"Timestamp {timestamp}, Total PNL ended up being {totpnl}")
        # print(f'Will trade {result}')
        logger.print("End transmission")
        conversions = 0
        trader_data = ""
        # del result['STARFRUIT']
        logger.flush(state, result,conversions,trader_data)
        return result, conversions, trader_data