"""
Graph Factory.
"""
from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.ERGraph import ERGraph
from Core.Graph.PassageGraph import PassageGraph
from Core.Graph.TreeGraph import TreeGraph
from Core.Graph.TreeGraphBalanced import TreeGraphBalanced
from Core.Graph.RKGraph import RKGraph
from Core.Graph.HKGraphTree import HKGraphTree
from Core.Graph.HKGraphTreeDynamic import HKGraphTreeDynamic


class GraphFactory():
    def __init__(self):
        self.creators = {
            "er_graph": self._create_er_graph,
            "rkg_graph": self._create_rkg_graph,
            "tree_graph": self._create_tree_graph,
            "tree_graph_balanced": self._create_tree_graph_balanced,
            "passage_graph": self._crease_passage_graph,
            "hk_graph_tree": self._create_hk_graph_tree,
            "hk_graph_tree_dynamic": self._create_hk_graph_tree_dynamic
        }


    def get_graph(self, config, **kwargs) -> BaseGraph:
        """Key is PersistType."""
        return self.creators[config.graph.graph_type](config, **kwargs)

    @staticmethod
    def _create_er_graph(config, **kwargs):
        return ERGraph(
            config.graph, **kwargs
        )

    @staticmethod
    def _create_rkg_graph(config, **kwargs):
        return RKGraph(config.graph, **kwargs)

    @staticmethod
    def _create_tree_graph(config, **kwargs):
        return TreeGraph(config, **kwargs)

    @staticmethod
    def _create_tree_graph_balanced(config, **kwargs):
        return TreeGraphBalanced(config, **kwargs)

    @staticmethod
    def _crease_passage_graph(config, **kwargs):
        return PassageGraph(config.graph, **kwargs)
    
    @staticmethod
    def _create_hk_graph_tree(config, **kwargs):
        return HKGraphTree(config.graph, config, **kwargs)
    
    @staticmethod
    def _create_hk_graph_tree_dynamic(config, **kwargs):
        return HKGraphTreeDynamic(config.graph, config, **kwargs)


get_graph = GraphFactory().get_graph
