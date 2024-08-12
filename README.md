# Curvetopia: A Journey into the World of Curves

**Date:** July 15, 2024

## Introduction

Welcome to Curvetopia, where we bring order and beauty to the world of 2D curves! This project guides you through identifying, regularizing, and beautifying various types of curves. Our mission is to develop an end-to-end process that transforms a raster image of line art into a set of smooth and well-defined curves, specifically using cubic Bézier curves. 

## Objectives

1. **Regularize Curves**: Identify and classify regular shapes within a given set of curves. The focus will be on:
   - Straight lines
   - Circles and ellipses
   - Rectangles and rounded rectangles
   - Regular polygons
   - Star shapes

2. **Explore Symmetry in Curves**: Analyze closed shapes for symmetry, particularly reflection symmetries, and transform the presentation into sets of points.

3. **Complete Incomplete Curves**: Develop algorithms to naturally complete 2D curves that have gaps or partial occlusions, ensuring smoothness and regularity in the process.

## Problem Description

We aim to convert line art images into polylines and then transform these into cubic Bézier curves. The input to our problem is a finite subset of paths, defined as sequences of points in 2D space. The output is a regularized and symmetrical set of paths represented as Bézier curves.

## Methodology

1. **Curve Regularization**: 
   - Identify and regularize straight lines, circles, ellipses, rectangles, rounded rectangles, polygons, and star shapes.
   - Test algorithms with diverse images to ensure accurate shape detection.

2. **Symmetry Exploration**: 
   - Detect reflection symmetries in closed shapes.
   - Transform Bézier curves into sets of points to identify symmetrical properties.

3. **Curve Completion**:
   - Develop computer vision techniques to fill gaps in curves due to occlusion removal.
   - Categorize shapes based on the level of occlusion: fully contained, partially contained, and disconnected.

## Implementation

The implementation uses CSV files to store polylines and Python code to read, visualize, and process these shapes. The project also includes algorithms for rasterizing SVG outputs to evaluate the correctness of solutions.

## Expected Results

- **Regularization and Symmetry**: The number of detected regular geometric shapes and their symmetries.
- **Occlusion Handling**: Evaluation through rasterization, comparing filled paths against expected outcomes.

## Usage

To visualize and process shapes, use the provided Python scripts. Load CSV files, apply regularization and symmetry detection algorithms, and visualize the results with the plotting functions.
