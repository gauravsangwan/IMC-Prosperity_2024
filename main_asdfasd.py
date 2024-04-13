# Following template is for imc algo visualizer 


import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
from datamodel import *

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
from typing import List
class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        trader_data  = ''
        conversions = 0
        time = state.timestamp
        if time == 0:
            od = state.order_depths['ORCHIDS']
            buy_orders = list(od.buy_orders.items())
            buy_orders.sort(key = lambda x:x[0], reverse = True)
            sell_orders = list(od.sell_orders.items())
            sell_orders.sort(key = lambda x: x[0])
            best_bid = buy_orders[0][0]
            best_ask = sell_orders[0][0]
            result['ORCHIDS'] = [Order('ORCHIDS',best_bid,-2)]
            logger.print(state.observations.conversionObservations['ORCHIDS'].bidPrice)
            logger.print(state.observations.conversionObservations['ORCHIDS'].askPrice)
            logger.print(state.observations.conversionObservations['ORCHIDS'].importTariff)
            logger.print(state.observations.conversionObservations['ORCHIDS'].exportTariff)
            logger.print(state.observations.conversionObservations['ORCHIDS'].transportFees)
        if time == 100:
            conversions = 1
            logger.print(state.observations.conversionObservations['ORCHIDS'].bidPrice)
            logger.print(state.observations.conversionObservations['ORCHIDS'].askPrice)
            logger.print(state.observations.conversionObservations['ORCHIDS'].importTariff)
            print(state.observations.conversionObservations['ORCHIDS'].exportTariff)
            print(state.observations.conversionObservations['ORCHIDS'].transportFees)
        # return result, conversions, trader_data
        # pnl = qty*( local best_bid at ts 0) - qty*(conversion best_ask at ts = 100) - qty*(import tariff) - qty*(transport fees)
        # pnl = 2*1094 - 2*1099 - 2*(-5) - 2*(0.9) = -1.8 calculation explanation
        


        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data