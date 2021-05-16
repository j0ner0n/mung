import os
import unittest
import numpy as np

from mung.io import read_nodes_from_file, export_node_list
from mung.node import Node


class NodeTest(unittest.TestCase):
    def test_bbox_to_integer_bounds(self):
        # Arrange
        expected = (44, 18, 56, 93)
        expected2 = (44, 18, 56, 93)

        # Act
        actual = Node.round_bounding_box_to_integer(44.2, 18.9, 55.1, 92.99)
        actual2 = Node.round_bounding_box_to_integer(44, 18, 56, 92.99)

        # Assert
        self.assertEqual(actual, expected)
        self.assertEqual(actual2, expected2)

    def test_overlaps(self):
        # Arrange
        node = Node(0, 'test', 10, 100, height=20, width=10)

        # Act and Assert
        self.assertEqual(node.bounding_box, (10, 100, 30, 110))

        self.assertTrue(node.overlaps((10, 100, 30, 110)))  # Exact match

        self.assertFalse(node.overlaps((0, 100, 8, 110)))  # Row mismatch
        self.assertFalse(node.overlaps((10, 0, 30, 89)))  # Column mismatch
        self.assertFalse(node.overlaps((0, 0, 8, 89)))  # Total mismatch

        self.assertTrue(node.overlaps((9, 99, 31, 111)))  # Encompasses Node
        self.assertTrue(node.overlaps((11, 101, 29, 109)))  # Within Node
        self.assertTrue(node.overlaps((9, 101, 31, 109)))  # Encompass horz., within vert.
        self.assertTrue(node.overlaps((11, 99, 29, 111)))  # Encompasses vert., within horz.
        self.assertTrue(node.overlaps((11, 101, 31, 111)))  # Corner within: top left
        self.assertTrue(node.overlaps((11, 99, 31, 109)))  # Corner within: top right
        self.assertTrue(node.overlaps((9, 101, 29, 111)))  # Corner within: bottom left
        self.assertTrue(node.overlaps((9, 99, 29, 109)))  # Corner within: bottom right

    def test_read_nodes_from_file(self):
        test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test', 'test_data')
        clfile = os.path.join(test_data_dir, '01_basic.xml')
        nodes = read_nodes_from_file(clfile)
        self.assertEqual(len(nodes), 48)

    def test_read_nodes_from_file_with_data(self):
        test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test', 'test_data')
        file = os.path.join(test_data_dir, '01_basic_binary_2.0.xml')
        nodes = read_nodes_from_file(file)
        self.assertEqual("G", nodes[0].data['pitch_step'])
        self.assertEqual(79, nodes[0].data['midi_pitch_code'])
        self.assertEqual([8, 17], nodes[0].data['precedence_outlinks'])
        
    def test_join_likelihoods(self):
        """Ensure that links and classes with their likelihoods are accurately stored when nodes are joined
        """
        node1 = Node(0, 'test1', 10, 10, 20, 20, [0.9, 0.05, 0, 0, 0, 0, 0.04], [2], [0.85], [3], [0.7], np.zeros((20, 20)))
        node2 = Node(1, 'test2', 15, 15, 30, 30, [0.1, 0.7, 0.1, 0.05, 0.05, 0, 0], [4], [0.5], [5], [0.5], np.zeros((30, 30)))
        node1.join(node2)
        node_accurate_join = Node(0, 'test1', 10, 10, 35, 35, [0.9, 0.05, 0, 0, 0, 0, 0.04], [2, 4], [0.85, 0.5], [3, 5], [0.7, 0.5], np.zeros((35,35)))
        self.assertEqual(node1.class_likelihoods, node_accurate_join.class_likelihoods)
        self.assertEqual(node1.inlinks, node_accurate_join.inlinks)
        self.assertEqual(node1.inlinks_likelihoods, node_accurate_join.inlinks_likelihoods)
        self.assertEqual(node1.outlinks, node_accurate_join.outlinks)
        self.assertEqual(node1.outlinks_likelihoods, node_accurate_join.outlinks_likelihoods)
        
    def test_xml_export(self):
        """Ensure that XML export of node works as expected with 3-digit floating point rounding performed for probability values
        """
        node = Node(0, 'test1', 10, 10, 20, 20, [1/7, 4/7, 1/7, 0, 0, 0, 1/7], [2], [0.85], [3], [0.7], np.zeros((20, 20)))
        expected = \
            """<Node>
	<Id>0</Id>
	<ClassName>test1</ClassName>
	<ClassLikelihoods>0.143 0.571 0.143 0.000 0.000 0.000 0.143</ClassLikelihoods>
	<Top>10</Top>
	<Left>10</Left>
	<Width>20</Width>
	<Height>20</Height>
	<Mask>0:400</Mask>
	<Inlinks>3</Inlinks>
	<InlinksLikelihoods>0.700</InlinksLikelihoods>
	<Outlinks>2</Outlinks>
	<OutlinksLikelihoods>0.850</OutlinksLikelihoods>
</Node>"""
        self.assertEqual(str(node), expected)
        


if __name__ == '__main__':
    unittest.main()
