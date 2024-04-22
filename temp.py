import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, OrderedDict
import collections 
from collections import defaultdict
import random
import math
import copy
import numpy as np


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


MAX_INT = int(1e9)

empty_dict = {'AMETHYSTS':0,'STARFRUIT':0, 'ORCHIDS':0,'CHOCOLATE':0,'STRAWBERRIES':0,'ROSES':0,'GIFT_BASKET':0,'COCONUT': 0, 'COCONUT_COUPON': 0}

class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100,'CHOCOLATE':250,'STRAWBERRIES':350,'ROSES':60,'GIFT_BASKET':60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
    PRODS = ['AMETHYSTS','STARFRUIT','ORCHIDS','CHOCOLATE','STRAWBERRIES','ROSES','GIFT_BASKET', 'COCONUT_COUPON','COCONUT']
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(default_value)
    person_actvalof_position = defaultdict(default_value)

    cpnl = defaultdict(lambda: 0)

    startfruit_cache = []
    startfruit_dim = 4
    steps = 0

    ORCHID_MM_RANGE = 5
    DIFFERENCE_MEAN = 379.4904833333333
    DIFFERENCE_STD = 76.42438217375009
    PERCENT_OF_STD_TO_TRADE_AT = 0.5


    ITERS = 30_000
    COUPON_DIFFERENCE_STD = 13.381768062409492
    COCONUT_DIFFERENCE_STD = 88.75266514702373
    PREV_COCONUT_PRICE = -1
    PREV_COUPON_PRICE = -1
    COCONUT_MEAN = 9999.900983333333
    COCONUT_SUM = 299997029.5
    COUPON_SUM = 19051393.0
    COUPON_Z_SCORE = 0.5

    COCONUT_STORE = []
    COCONUT_STORE_SIZE = 25
    COCONUT_BS_STORE = []

    delta_signs = 1
    time = 0

    COUPON_IV_STORE = []
    COUPON_IV_STORE_SIZE = 25




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
    
    def black_scholes(self, S, K, T, r, sigma, mean):
        def N(x):
        # return sp.stats.norm.cdf(x, mean, std**2)
            return 0.5 * (1 + math.erf((x-mean) / (sigma**2*math.sqrt(2))))
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r*T)* N(d2)

    def black_scholes_price(self, S, K, t, r, sigma):
        def cdf(x):
            return 0.5 * (1 + math.erf(x/math.sqrt(2)))

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        price = S * cdf(d1) - K * np.exp(-r * t) * cdf(d2)
        return price

    def newtons_method(self, f, x0=0.02, epsilon=1e-7, max_iter=100, h=1e-5):
        def numerical_derivative(f, x, h=1e-5):
            return (f(x + h) - f(x - h)) / (2 * h)
        
        x = x0
        for i in range(max_iter):
            fx = f(x)
            if abs(fx) < epsilon:
                return x
            dfx = numerical_derivative(f, x, h)
            if dfx == 0:
                raise ValueError("Derivative zero. No solution found.")
            x -= fx / dfx
        raise ValueError("Maximum iterations reached. No solution found.")
    
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
    
    def calculate_orders(self,product, order_depth, our_bid, our_ask, orchid = False):
        orders: list[Order] = []
        
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        logger.print(f'Product: {product} - best sell: {best_sell_price}, best buy: {best_buy_price}')

        position1 = self.position[product] if not orchid else 0
        limit = self.POSITION_LIMIT[product]

        # penny the current highest bid / lowest ask 
        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        bid_price = min(penny_buy, our_bid)
        ask_price = max(penny_sell, our_ask)

        if orchid:
            ask_price = max(best_buy_price+3, our_ask)

        # MARKET TAKE ASKS (buy items)
        for ask, vol in sell_orders.items():
            if position1 < limit and (ask <= our_bid or (position1 < 0 and ask == our_bid+1)): 
                num_orders = min(-vol, limit - position1)
                position1 += num_orders
                orders.append(Order(product, ask, num_orders))

        # MARKET MAKE BY PENNYING
        if position1 < limit:
            num_orders = limit - position1
            orders.append(Order(product, bid_price, num_orders))
            position1 += num_orders

        # RESET POSITION
        position1 = self.position[product] if not orchid else 0

        # MARKET TAKE BIDS (sell items)
        for bid, vol in buy_orders.items():
            if position1 > -limit and (bid >= our_ask or (position1 > 0 and bid+1 == our_ask)):
                num_orders = max(-vol, -limit-position1)
                position1 += num_orders
                orders.append(Order(product, bid, num_orders))

        # MARKET MAKE BY PENNYING
        if position1 > -limit:
            num_orders = -limit - position1
            orders.append(Order(product, ask_price, num_orders))
            position1 += num_orders 

        return orders
    
    def get_volume_and_best_price(self,orders,buy_order):
        volume = 0
        best = 0 if buy_order else MAX_INT

        for price, vol in orders.items():
            if buy_order:
                volume += vol
                best = max(best,price)
            else:
                volume -= vol
                best = min(best,price)
        return volume, best
    
    def compute_orders(self, product, order_depth, acc_bid, acc_ask):

        if product == "AMETHYSTS":
            return self.compute_orders_amethyst(product, order_depth, acc_bid, acc_ask)
        
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS':[],'CHOCOLATE': [],'STRAWBERRIES':[],'ROSES':[],'GIFT_BASKET':[]}
        result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS':[],'CHOCOLATE': [],'STRAWBERRIES':[],'ROSES':[],'GIFT_BASKET':[],'COCONUT_COUPON':[],'COCONUT':[]}

        for key, val in state.position.items():
            self.position[key] = val
        for key, val in self.position.items():
            logger.print(f'{key} position: {val}')
        
        timestamp = state.timestamp

        # if len(self.startfruit_cache) == self.startfruit_dim:
        #     self.startfruit_cache.pop(0)

        # _, bs_starfruit = self.extract_values(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        # _, bb_starfruit = self.extract_values(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        # self.startfruit_cache.append((bs_starfruit + bb_starfruit)/2)

        # starfruit_lb = -MAX_INT
        # starfruit_ub = MAX_INT

        # if len(self.startfruit_cache) == self.startfruit_dim:
        #     starfruit_lb = self.calc_next_price_starfruit() - 1
        #     starfruit_ub = self.calc_next_price_starfruit() + 1
        
        # amethyst_lb = 10000
        # amethyst_ub = 10000

        # acc_bid = {'AMETHYSTS': amethyst_lb, 'STARFRUIT': starfruit_lb}
        # acc_ask = {'AMETHYSTS': amethyst_ub, 'STARFRUIT': starfruit_ub}

        self.steps +=1 

        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                self.person_position[trade.buyer][product] =1.5
                self.person_position[trade.seller][product] = -1.5
                self.person_actvalof_position[trade.buyer][product] += trade.quantity
                self.person_actvalof_position[trade.seller][product] += -trade.quantity
        
        for product in self.PRODS:
            # if product == 'AMETHYSTS' or product == 'STARFRUIT':
            #     order_depth: OrderDepth = state.order_depths[product]
            #     orders = self.compute_orders(product,order_depth,acc_bid[product],acc_ask[product])
            #     result[product] += orders
            # elif product == 'ORCHIDS' :
            #     order_depth: OrderDepth = state.order_depths[product]
            #     orders: list[Order] = []
            #     shipping_cost = state.observations.conversionObservations['ORCHIDS'].transportFees
            #     import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
            #     export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
            #     ducks_ask = state.observations.conversionObservations['ORCHIDS'].askPrice
            #     ducks_bid = state.observations.conversionObservations['ORCHIDS'].bidPrice
            #     buy_from_ducks_prices = ducks_ask + shipping_cost + import_tariff
            #     sell_to_ducks_prices = ducks_bid + shipping_cost + export_tariff

            #     orchids_lb = int(round(buy_from_ducks_prices))-1
            #     orchids_ub = int(round(buy_from_ducks_prices))+1
            #     orders += self.calculate_orders(product, order_depth, orchids_lb,orchids_ub, orchid=True)
            #     conversions = -self.position[product]

            #     # logger.print(f'buying from ducks for: {buy_from_ducks_prices}')
            #     # logger.print(f'selling to ducks for: {sell_to_ducks_prices}')
            #     result[product] += orders
            # elif product == 'GIFT_BASKET':
            #     order_depth: OrderDepth = state.order_depths[product]
            #     orders: list[Order] = []
            #     _, choco_best_sell_price = self.get_volume_and_best_price(state.order_depths['CHOCOLATE'].sell_orders, buy_order=False)
            #     _, choco_best_buy_price = self.get_volume_and_best_price(state.order_depths['CHOCOLATE'].buy_orders, buy_order=True)
            #     _, straw_best_sell_price = self.get_volume_and_best_price(state.order_depths['STRAWBERRIES'].sell_orders, buy_order=False)
            #     _, straw_best_buy_price = self.get_volume_and_best_price(state.order_depths['STRAWBERRIES'].buy_orders, buy_order=True)
            #     _, roses_best_sell_price = self.get_volume_and_best_price(state.order_depths['ROSES'].sell_orders, buy_order=False)
            #     _, roses_best_buy_price = self.get_volume_and_best_price(state.order_depths['ROSES'].buy_orders, buy_order=True)

            #     basket_items = ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']
            #     mid_price = {}
            #     for item in basket_items:
            #         _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
            #         _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders, buy_order=True)

            #         mid_price[item] = (best_sell_price+best_buy_price)/2
            #     difference = mid_price['GIFT_BASKET'] - 4*mid_price['CHOCOLATE'] - 6*mid_price['STRAWBERRIES'] - mid_price['ROSES'] - self.DIFFERENCE_MEAN
            #     logger.print(f'For basket, difference: {difference}')
            #     worst_bid_price = min(order_depth.buy_orders.keys())
            #     worst_ask_price = max(order_depth.sell_orders.keys())

            #     if difference > self.PERCENT_OF_STD_TO_TRADE_AT * self.DIFFERENCE_STD: # basket overvalued, sell
            #         orders += self.calculate_orders(product, order_depth, -MAX_INT, worst_bid_price)
                
            #     elif difference < -self.PERCENT_OF_STD_TO_TRADE_AT * self.DIFFERENCE_STD: # basket undervalued, buy
            #         orders += self.calculate_orders(product, order_depth, worst_ask_price, MAX_INT)
            #     # logger.print("ORDERS",orders)
            #     result[product] += orders
            if product == 'COCONUT_COUPON':
                # if product == 'COCONUT_COUPON':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []

                items = ['COCONUT', 'COCONUT_COUPON']
                mid_price, best_bid_price, best_ask_price = {}, {}, {}
                for item in items : 
                    _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
                    _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders, buy_order=True)

                    mid_price[item] = (best_sell_price+best_buy_price)/2
                    best_bid_price[item] = best_buy_price
                    best_ask_price[item] = best_sell_price

                self.COCONUT_SUM += mid_price['COCONUT']
                self.COUPON_SUM += mid_price['COCONUT_COUPON']
                self.ITERS += 1
                coconut_mean = self.COCONUT_SUM/self.ITERS
                coupon_mean = self.COUPON_SUM/self.ITERS

                self.COCONUT_STORE.append(mid_price['COCONUT'])
                store = np.array(self.COCONUT_STORE)
                mean, std = np.mean(store), np.std(store)

                iv = self.newtons_method(lambda sigma: self.black_scholes_price(mid_price['COCONUT'], 10_000, 250, 0, sigma) - mid_price['COCONUT_COUPON'])
                logger.print("IV",iv)
                self.COUPON_IV_STORE.append(iv)
                # bs_std = np.std(self.COCONUT_BS_STORE)


                # predicted_coupon_price = (mid_price['COCONUT'] - coconut_mean) * self.COUPON_Z_SCORE + coupon_mean
                # predicted_coupon_price = (mid_price['COCONUT'] - coconut_mean) * 0.503+ coupon_mean

                # logger.print(f'For coconut coupon, predicted price: {predicted_coupon_price}')
                # difference = mid_price['COCONUT_COUPON'] - predicted_coupon_price


                # if difference > self.COUPON_Z_SCORE * self.COUPON_DIFFERENCE_STD: # coupon overvalued, sell
                #     orders += self.calculate_orders(product, order_depth, -MAX_INT, best_bid_price['COCONUT_COUPON'])
                
                # elif difference < -self.COUPON_Z_SCORE * self.COUPON_DIFFERENCE_STD:
                #     orders += self.calculate_orders(product, order_depth, best_ask_price['COCONUT_COUPON'], MAX_INT)
                # result[product] += orders


                if len(self.COUPON_IV_STORE) >= self.COUPON_IV_STORE_SIZE:
                    # coco_price = mid_price['COCONUT']
                    # logger.print(f'coconut price: {coco_price}, mean: {mean}, std: {std}')
                    # logger.print(f'predicted coupon price: {curr_bs_est}')
                    # difference = mid_price['COCONUT_COUPON'] - curr_bs_est

                    iv_mean, iv_std = np.mean(self.COUPON_IV_STORE), np.std(self.COUPON_IV_STORE)

                    difference = iv - iv_mean

                    if difference > self.COUPON_Z_SCORE * iv_std:
                        # iv too high, overpriced, sell
                        orders += self.calculate_orders(product, order_depth, -MAX_INT, best_bid_price['COCONUT_COUPON'])
                        
                    elif difference < -self.COUPON_Z_SCORE * iv_std:
                        # iv too low, underpriced, buy
                        orders += self.calculate_orders(product, order_depth, best_ask_price['COCONUT_COUPON'], MAX_INT)

                    # self.COCONUT_STORE.pop(0)
                    # self.COCONUT_BS_STORE.pop(0)

                    self.COUPON_IV_STORE.pop(0)
                
                self.PREV_COCONUT_PRICE = mid_price['COCONUT']
                self.PREV_COUPON_PRICE = mid_price['COCONUT_COUPON']
                result[product] += orders


            # logger.print(f'placed orders: {orders}')     

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