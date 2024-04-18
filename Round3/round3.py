import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, OrderedDict, Dict
import collections 
from collections import defaultdict
import random
import math
import copy
import numpy as np
import jsonpickle

MAX_INT = int(1e9)

empty_dict = {'AMETHYSTS':0,'STARFRUIT':0, 'ORCHIDS':0,'CHOCOLATE':0,'STRAWBERRIES':0,'ROSES':0,'GIFT_BASKET':0}
class HistoricalVWAP:
    def __init__(self, bv=0, sv=0, bpv=0, spv=0):
        self.buy_volume = bv
        self.sell_volume = sv
        self.buy_price_volume = bpv
        self.sell_price_volume = spv
class RecordedData: 
    def __init__(self):
        self.amethyst_hvwap = HistoricalVWAP()
        self.starfruit_cache = []
        self.LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100,'CHOCOLATE':250,'STRAWBERRIES':350,'ROSES':60,'GIFT_BASKET':60}
        self.INF = int(1e9)
        self.STARFRUIT_CACHE_SIZE = 38
        self.AME_RANGE = 2
        self.POSITION = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS': 0,'CHOCOLATE':0,'STRAWBERRIES':0,'ROSES':0,'GIFT_BASKET':0}
        self.ORCHID_MM_RANGE = 5
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
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS': 100,'CHOCOLATE':250,'STRAWBERRIES':350,'ROSES':60,'GIFT_BASKET':60}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(default_value)
    person_actvalof_position = defaultdict(default_value)

    cpnl = defaultdict(lambda: 0)

    startfruit_cache = []
    startfruit_dim = 4
    steps = 0
    premium_basket = 0
    ORCHID_MM_RANGE = 5
    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0
    std = 21
    basket_std = 372
    basket_prev = None
    dip_prev = None
    ukulele_prev = None
    etf_returns = np.array([])
    asset_returns = np.array([])


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
    
        # gets the total traded volume of each time stamp and best price
    # best price in buy_orders is the max; best price in sell_orders is the min
    # buy_order indicates orders are buy or sell orders
    def get_volume_and_best_price(self, orders, buy_order):
        volume = 0
        best = 0 if buy_order else self.INF

        for price, vol in orders.items():
            if buy_order:
                volume += vol
                best = max(best, price)
            else:
                volume -= vol
                best = min(best, price)

        return volume, best

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
        

    def calculate_orders(self, product, order_depth, our_bid, our_ask, orchild=False):
        orders: list[Order] = []
        
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        logger.print(f'Product: {product} - best sell: {best_sell_price}, best buy: {best_buy_price}')

        position = self.POSITION[product] if not orchild else 0
        limit = self.LIMIT[product]

        # penny the current highest bid / lowest ask 
        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        bid_price = min(penny_buy, our_bid)
        ask_price = max(penny_sell, our_ask)

        if orchild:
            ask_price = max(best_buy_price+3, our_ask)

        # MARKET TAKE ASKS (buy items)
        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid+1)): 
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        # MARKET MAKE BY PENNYING
        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        # RESET POSITION
        position = self.POSITION[product] if not orchild else 0

        # MARKET TAKE BIDS (sell items)
        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= our_ask or (position > 0 and bid+1 == our_ask)):
                num_orders = max(-vol, -limit-position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        # MARKET MAKE BY PENNYING
        if position > -limit:
            num_orders = -limit - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders 

        return orders

    def compute_orders_basket(self, order_depth):
        orders = {'GIFT_BASKET': [], 'STRAWBERRIES': [], 'CHOCOLATE': [], 'ROSES': []}
        prods = ['GIFT_BASKET', 'STRAWBERRIES', 'CHOCOLATE', 'ROSES']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break
        #Residual buy and sell
        res_buy = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*4 - mid_price['CHOCOLATE']*2 - mid_price['ROSES'] - self.premium_basket
        res_sell = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*4 - mid_price['CHOCOLATE']*2 - mid_price['ROSES'] - self.premium_basket

        trade_at = self.basket_std*0.5
        close_at = self.basket_std*(-1000)

        pb_pos = self.position['GIFT_BASKET']
        pb_neg = self.position['GIFT_BASKET']

        roses_pos = self.position['ROSES']
        roses_neg = self.position['ROSES']


        basket_buy_sig = 0
        basket_sell_sig = 0

        if self.position['GIFT_BASKET'] == self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_buy_basket_unfill = 0
        if self.position['GIFT_BASKET'] == -self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_sell_basket_unfill = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            self.cont_buy_basket_unfill = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                self.cont_sell_basket_unfill += 2
                pb_neg -= vol
                #roses_pos += vol
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                self.cont_buy_basket_unfill += 2
                pb_pos += vol

        # logger.print("OLIVIA_ROSES",self.person_position)
        if int(round(self.person_position['Olivia']['ROSES'])) > 0:
            val_ord = self.POSITION_LIMIT['ROSES'] - roses_pos
            if val_ord > 0:
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], val_ord))
        if int(round(self.person_position['Olivia']['ROSES'])) < 0:
            val_ord = -(self.POSITION_LIMIT['ROSES'] + roses_neg)
            if val_ord < 0:
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], val_ord))



        return orders


    def compute_orders_basket2(self, order_depth, state): 
        product = 'GIFT_BASKET' 
        orders_GIFT_BASKET = list[Order]()
        # current positions
        basket_pos = state.position.get("GIFT_BASKET", 0)
        CHOCOLATE_pos = state.position.get("CHOCOLATE", 0)
        STRAWBERRIES_pos = state.position.get("STRAWBERRIES", 0)
        ROSES_pos = state.position.get("ROSES", 0)

    ##################################################################################
        
        basket_buy_orders: Dict[int, int] = state.order_depths[product].buy_orders
        basket_sell_orders: Dict[int, int] = state.order_depths[product].sell_orders

        basket_best_bid: float = max(basket_buy_orders)
        basket_best_ask: float = min(basket_sell_orders)

        # Finding price / NAV ratio
        basket_price: float = (basket_best_bid + basket_best_ask) / 2

        CHOCOLATE_buy_orders: Dict[int, int] = state.order_depths['CHOCOLATE'].buy_orders
        CHOCOLATE_sell_orders: Dict[int, int] = state.order_depths['CHOCOLATE'].sell_orders

        CHOCOLATE_best_bid: float = max(CHOCOLATE_buy_orders)
        CHOCOLATE_best_ask: float = min(CHOCOLATE_sell_orders)

        CHOCOLATE_price: float = (CHOCOLATE_best_bid + CHOCOLATE_best_ask) / 2

        STRAWBERRIES_buy_orders: Dict[int, int] = state.order_depths['STRAWBERRIES'].buy_orders
        STRAWBERRIES_sell_orders: Dict[int, int] = state.order_depths['STRAWBERRIES'].sell_orders

        STRAWBERRIES_best_bid: float = max(STRAWBERRIES_buy_orders)
        STRAWBERRIES_best_ask: float = min(STRAWBERRIES_sell_orders)

        STRAWBERRIES_price: float = (STRAWBERRIES_best_bid + STRAWBERRIES_best_ask) / 2

        ROSES_buy_orders: Dict[int, int] = state.order_depths['ROSES'].buy_orders
        ROSES_sell_orders: Dict[int, int] = state.order_depths['ROSES'].sell_orders

        ROSES_best_bid: float = max(ROSES_buy_orders)
        ROSES_best_ask: float = min(ROSES_sell_orders)

        ROSES_price: float = (ROSES_best_bid + ROSES_best_ask) / 2

        est_price: float = 6 * STRAWBERRIES_price + 4 * CHOCOLATE_price + ROSES_price

        price_nav_ratio: float = basket_price / est_price

    ##################################################################################

        self.etf_returns = np.append(self.etf_returns, basket_price)
        self.asset_returns = np.append(self.asset_returns, est_price)

        rolling_mean_etf = np.mean(self.etf_returns[-10:])
        rolling_std_etf = np.std(self.etf_returns[-10:])

        rolling_mean_asset = np.mean(self.asset_returns[-10:])
        rolling_std_asset = np.std(self.asset_returns[-10:])

        z_score_etf = (self.etf_returns[-1] - rolling_mean_etf) / rolling_std_etf
        z_score_asset = (self.asset_returns[-1] - rolling_mean_asset) / rolling_std_asset

        z_score_diff = z_score_etf - z_score_asset

        print(f'ZSCORE DIFF = {z_score_diff}')

        # implement stop loss
        # stop_loss = 0.01

        #if price_nav_ratio < self.basket_pnav_ratio - self.basket_eps:
        if z_score_diff < -2:
            # stop_loss_price = self.etf_returns[-2] 


            # ETF is undervalued! -> we buy ETF and sell individual assets!
            # Finds volume to buy that is within position limit
            #basket_best_ask_vol = max(basket_pos-self.basket_limit, state.order_depths['GIFT_BASKET'].sell_orders[basket_best_ask])
            basket_best_ask_vol = state.order_depths['GIFT_BASKET'].sell_orders[basket_best_ask]
            CHOCOLATE_best_bid_vol =  state.order_depths['CHOCOLATE'].buy_orders[CHOCOLATE_best_bid]
            STRAWBERRIES_best_bid_vol = state.order_depths['STRAWBERRIES'].buy_orders[STRAWBERRIES_best_bid]
            ROSES_best_bid_vol = state.order_depths['ROSES'].buy_orders[ROSES_best_bid]

            limit_mult = min(-basket_best_ask_vol, ROSES_best_bid_vol, 
                                round(CHOCOLATE_best_bid_vol / 2), round(STRAWBERRIES_best_bid_vol / 4))

            print(f'LIMIT: {limit_mult}')

            print("BUY", 'GIFT_BASKET', limit_mult, "x", basket_best_ask)
            orders_GIFT_BASKET.append(Order('GIFT_BASKET', basket_best_ask, limit_mult))
            
        #elif price_nav_ratio > self.basket_pnav_ratio + self.basket_eps:
        elif z_score_diff > 2:
            # ETF is overvalued! -> we sell ETF and buy individual assets!
            # Finds volume to buy that is within position limit
            #basket_best_bid_vol = min(self.basket_limit-basket_pos, state.order_depths['GIFT_BASKET'].buy_orders[basket_best_bid])
            basket_best_bid_vol = state.order_depths['GIFT_BASKET'].buy_orders[basket_best_bid]
            CHOCOLATE_best_ask_vol = state.order_depths['CHOCOLATE'].sell_orders[CHOCOLATE_best_ask]
            STRAWBERRIES_best_ask_vol = state.order_depths['STRAWBERRIES'].sell_orders[STRAWBERRIES_best_ask]
            ROSES_best_ask_vol = state.order_depths['ROSES'].sell_orders[ROSES_best_ask]

            limit_mult = min(basket_best_bid_vol, -ROSES_best_ask_vol, 
                                round(-CHOCOLATE_best_ask_vol / 2), round(-STRAWBERRIES_best_ask_vol / 4))

            print(f'LIMIT: {limit_mult}')

            print("SELL", 'GIFT_BASKET', limit_mult, "x", basket_best_bid)
            orders_GIFT_BASKET.append(Order('GIFT_BASKET', basket_best_bid, -limit_mult))
        return orders_GIFT_BASKET
    

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {'AMETHYSTS': [], 'STARFRUIT': [], 'ORCHIDS':[],'CHOCOLATE': [],'STRAWBERRIES':[],'ROSES':[],'GIFT_BASKET':[]}
        for key, val in state.position.items():
            self.position[key] = val
        for key, val in self.position.items():
            logger.print(f'{key} position: {val}')
        assert abs(self.position.get('ROSES', 0)) <= self.POSITION_LIMIT['ROSES']
        conversions = 0
        timestamp = state.timestamp
        if state.traderData == '': # first run, set up data
            data = RecordedData()
        else:
            data = jsonpickle.decode(state.traderData)

        self.LIMIT = data.LIMIT
        self.INF = data.INF
        self.STARFRUIT_CACHE_SIZE = data.STARFRUIT_CACHE_SIZE
        self.AME_RANGE = data.AME_RANGE
        self.POSITION = data.POSITION
        self.ORCHID_MM_RANGE = data.ORCHID_MM_RANGE

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
                # if trade.buyer == trade.seller:
                #     continue
                self.person_position['Olivia'][product] =1.5
                self.person_position[trade.seller][product] = -1.5
                self.person_actvalof_position['Olivia'][product] += trade.quantity
                self.person_actvalof_position[trade.seller][product] += -trade.quantity
        
        for product in ['AMETHYSTS','STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product,order_depth,acc_bid[product],acc_ask[product])
            result[product] += orders
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            if product == 'ORCHIDS':
                shipping_cost = state.observations.conversionObservations['ORCHIDS'].transportFees
                import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
                export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
                ducks_ask = state.observations.conversionObservations['ORCHIDS'].askPrice
                ducks_bid = state.observations.conversionObservations['ORCHIDS'].bidPrice

                buy_from_ducks_prices = ducks_ask + shipping_cost + import_tariff
                sell_to_ducks_prices = ducks_bid + shipping_cost + export_tariff

                lower_bound = int(round(buy_from_ducks_prices))-1
                upper_bound = int(round(buy_from_ducks_prices))+1

                orders += self.calculate_orders(product, order_depth, lower_bound, upper_bound, orchild=True)
                conversions = -self.POSITION[product]

                logger.print(f'buying from ducks for: {buy_from_ducks_prices}')
                logger.print(f'selling to ducks for: {sell_to_ducks_prices}')
                result[product] += orders
        
        #ROund3

        orders = self.compute_orders_basket(state.order_depths)
        result['GIFT_BASKET'] += orders
        # result['GIFT_BASKET'] += orders['GIFT_BASKET']
        # result['STRAWBERRIES'] += orders['STRAWBERRIES']
        # result['CHOCOLATE'] += orders['CHOCOLATE']
        # result['ROSES'] += orders['R  OSES']


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

        for person in self.person_position.keys():
            for val in self.person_position[person].keys():
                
                if person == 'Olivia':
                    self.person_position[person][val] *= 0.995
                if person == 'Pablo':
                    self.person_position[person][val] *= 0.8
                if person == 'Camilla':
                    self.person_position[person][val] *= 0
        
        logger.print(f"Timestamp {timestamp}, Total PNL ended up being {totpnl}")
        # print(f'Will trade {result}')
        logger.print("End transmission")
        trader_data = jsonpickle.encode(data)
        # del result['STARFRUIT']
        logger.flush(state, result,conversions,trader_data)
        return result, conversions, trader_data