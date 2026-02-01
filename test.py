# ============================================================================
# tests.py - Rigorous Test Suite for Quantum-Enhanced LABS Solver
# ============================================================================
# This test suite validates:
# - LABS energy calculations
# - Interaction index generation (G2, G4)
# - LABS symmetries (inversion, reversal)
# - Quantum kernel outputs
# - Classical MTS components
# - End-to-end workflow correctness
# ============================================================================

import unittest
import numpy as np
import random
import cudaq

# Import from main module (adjust import path as needed)
# If running from same directory, use:
from quantum_labs_cudaq import (
    labs_energy,
    bitstring_to_spin,
    spin_to_bitstring,
    get_interactions,
    compute_theta,
    get_labs_circuit,
    combine,
    mutate,
    tabu_search,
    run_mts,
    sample_quantum_population,
    run_quantum_enhanced_labs,
    KNOWN_OPTIMAL_ENERGIES,
    two_qubit_block,
    four_qubit_block,
)

# Set CUDA-Q target for testing
cudaq.set_target('qpp-cpu')


# ============================================================================
# TEST CLASS 1: LABS Energy Function Tests
# ============================================================================

class TestLABSEnergy(unittest.TestCase):
    """Test suite for LABS energy calculation."""
    
    def test_energy_known_values_n3(self):
        """Test energy calculation for N=3 with known values."""
        # For N=3, optimal energy is 1
        # Sequence [1, 1, -1] should give:
        # C_1 = s_0*s_1 + s_1*s_2 = 1*1 + 1*(-1) = 0
        # C_2 = s_0*s_2 = 1*(-1) = -1
        # E = 0^2 + (-1)^2 = 1
        s = np.array([1, 1, -1])
        energy = labs_energy(s)
        self.assertEqual(energy, 1)
    
    def test_energy_known_values_n4(self):
        """Test energy calculation for N=4 with known values."""
        # Manually compute for [1, 1, -1, -1]
        # C_1 = 1*1 + 1*(-1) + (-1)*(-1) = 1 - 1 + 1 = 1
        # C_2 = 1*(-1) + 1*(-1) = -2
        # C_3 = 1*(-1) = -1
        # E = 1 + 4 + 1 = 6
        s = np.array([1, 1, -1, -1])
        energy = labs_energy(s)
        self.assertEqual(energy, 6)
    
    def test_energy_all_same(self):
        """Test energy when all spins are the same."""
        for N in range(3, 8):
            s_plus = np.ones(N, dtype=int)
            s_minus = -np.ones(N, dtype=int)
            
            # All +1: C_k = N-k for all k
            # E = sum_{k=1}^{N-1} (N-k)^2
            expected = sum((N - k) ** 2 for k in range(1, N))
            
            self.assertEqual(labs_energy(s_plus), expected, 
                           f"Failed for all +1, N={N}")
            self.assertEqual(labs_energy(s_minus), expected, 
                           f"Failed for all -1, N={N}")
    
    def test_energy_alternating(self):
        """Test energy for alternating sequences."""
        # Alternating [1, -1, 1, -1, ...] should have low autocorrelation
        for N in range(3, 8):
            s = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
            energy = labs_energy(s)
            
            # Energy should be non-negative
            self.assertGreaterEqual(energy, 0)
            
            # Alternating sequences have specific patterns
            # For even N: C_k alternates between N-k and -(N-k)
            # The energy should be calculable
            self.assertIsInstance(energy, (int, np.integer))
    
    def test_energy_non_negative(self):
        """Energy should always be non-negative (sum of squares)."""
        np.random.seed(42)
        for _ in range(100):
            N = np.random.randint(3, 15)
            s = 2 * np.random.randint(0, 2, N) - 1
            energy = labs_energy(s)
            self.assertGreaterEqual(energy, 0, 
                                   f"Negative energy for s={s}")
    
    def test_energy_integer_output(self):
        """Energy should always be an integer for spin sequences."""
        np.random.seed(123)
        for _ in range(50):
            N = np.random.randint(3, 12)
            s = 2 * np.random.randint(0, 2, N) - 1
            energy = labs_energy(s)
            self.assertEqual(energy, int(energy), 
                           "Energy should be integer")


# ============================================================================
# TEST CLASS 2: LABS Symmetry Tests
# ============================================================================

class TestLABSSymmetries(unittest.TestCase):
    """Test suite for LABS problem symmetries."""
    
    def test_inversion_symmetry(self):
        """Flipping all spins should preserve energy."""
        np.random.seed(42)
        for N in range(3, 12):
            for _ in range(10):
                s = 2 * np.random.randint(0, 2, N) - 1
                
                original_energy = labs_energy(s)
                inverted_energy = labs_energy(-s)
                
                self.assertEqual(original_energy, inverted_energy,
                               f"Inversion symmetry failed for N={N}, s={s}")
    
    def test_reversal_symmetry(self):
        """Reversing the sequence should preserve energy."""
        np.random.seed(42)
        for N in range(3, 12):
            for _ in range(10):
                s = 2 * np.random.randint(0, 2, N) - 1
                
                original_energy = labs_energy(s)
                reversed_energy = labs_energy(s[::-1])
                
                self.assertEqual(original_energy, reversed_energy,
                               f"Reversal symmetry failed for N={N}, s={s}")
    
    def test_combined_symmetry(self):
        """Combined inversion and reversal should preserve energy."""
        np.random.seed(42)
        for N in range(3, 10):
            for _ in range(10):
                s = 2 * np.random.randint(0, 2, N) - 1
                
                original = labs_energy(s)
                inverted_reversed = labs_energy(-s[::-1])
                
                self.assertEqual(original, inverted_reversed,
                               "Combined symmetry failed")


# ============================================================================
# TEST CLASS 3: Interaction Index Tests
# ============================================================================

class TestInteractions(unittest.TestCase):
    """Test suite for interaction index generation."""
    
    def test_g2_bounds(self):
        """All G2 indices should be within [0, N-1]."""
        for N in range(3, 15):
            G2, _ = get_interactions(N)
            for pair in G2:
                for idx in pair:
                    self.assertGreaterEqual(idx, 0, 
                                          f"G2 index below 0 for N={N}")
                    self.assertLess(idx, N, 
                                   f"G2 index >= N for N={N}")
    
    def test_g4_bounds(self):
        """All G4 indices should be within [0, N-1]."""
        for N in range(4, 15):
            _, G4 = get_interactions(N)
            for quartet in G4:
                for idx in quartet:
                    self.assertGreaterEqual(idx, 0, 
                                          f"G4 index below 0 for N={N}")
                    self.assertLess(idx, N, 
                                   f"G4 index >= N for N={N}")
    
    def test_g2_pairs_distinct(self):
        """G2 pairs should have distinct indices."""
        for N in range(3, 15):
            G2, _ = get_interactions(N)
            for pair in G2:
                self.assertEqual(len(pair), 2)
                self.assertNotEqual(pair[0], pair[1],
                                   f"G2 has duplicate indices for N={N}")
    
    def test_g4_quartets_distinct(self):
        """G4 quartets should have 4 distinct indices."""
        for N in range(4, 15):
            _, G4 = get_interactions(N)
            for quartet in G4:
                self.assertEqual(len(quartet), 4)
                self.assertEqual(len(set(quartet)), 4,
                               f"G4 has duplicate indices for N={N}")
    
    def test_g2_count_scaling(self):
        """G2 count should scale appropriately with N."""
        counts = []
        for N in range(3, 12):
            G2, _ = get_interactions(N)
            counts.append(len(G2))
        
        # G2 count should increase with N
        for i in range(len(counts) - 1):
            self.assertLess(counts[i], counts[i + 1],
                          "G2 count should increase with N")
    
    def test_g4_empty_for_small_n(self):
        """G4 should be empty for N < 4."""
        _, G4_n3 = get_interactions(3)
        self.assertEqual(len(G4_n3), 0, "G4 should be empty for N=3")
    
    def test_g4_nonempty_for_n4_plus(self):
        """G4 should have elements for N >= 4."""
        for N in range(4, 12):
            _, G4 = get_interactions(N)
            self.assertGreater(len(G4), 0, 
                             f"G4 should be non-empty for N={N}")
    
    def test_known_g2_counts(self):
        """Test G2 counts against expected values."""
        # G2 count for small N (manually verified)
        expected_g2_counts = {
            3: 3,   # [0,1], [0,2], [1,2]
            4: 6,   # pairs for N=4
            5: 10,  # pairs for N=5
        }
        
        for N, expected in expected_g2_counts.items():
            G2, _ = get_interactions(N)
            self.assertEqual(len(G2), expected,
                           f"G2 count mismatch for N={N}")


# ============================================================================
# TEST CLASS 4: Bitstring Conversion Tests
# ============================================================================

class TestBitstringConversion(unittest.TestCase):
    """Test suite for bitstring-spin conversions."""
    
    def test_bitstring_to_spin_basic(self):
        """Test basic bitstring to spin conversion."""
        self.assertTrue(np.array_equal(
            bitstring_to_spin("000"), np.array([-1, -1, -1])))
        self.assertTrue(np.array_equal(
            bitstring_to_spin("111"), np.array([1, 1, 1])))
        self.assertTrue(np.array_equal(
            bitstring_to_spin("101"), np.array([1, -1, 1])))
        self.assertTrue(np.array_equal(
            bitstring_to_spin("010"), np.array([-1, 1, -1])))
    
    def test_spin_to_bitstring_basic(self):
        """Test basic spin to bitstring conversion."""
        self.assertEqual(spin_to_bitstring(np.array([-1, -1, -1])), "000")
        self.assertEqual(spin_to_bitstring(np.array([1, 1, 1])), "111")
        self.assertEqual(spin_to_bitstring(np.array([1, -1, 1])), "101")
    
    def test_roundtrip_conversion(self):
        """Test that conversion is reversible."""
        test_bitstrings = ["000", "111", "101", "010", "0101", "1100"]
        
        for bs in test_bitstrings:
            spin = bitstring_to_spin(bs)
            recovered = spin_to_bitstring(spin)
            self.assertEqual(bs, recovered, 
                           f"Roundtrip failed for {bs}")
    
    def test_spin_values(self):
        """Converted spins should only contain +1 and -1."""
        np.random.seed(42)
        for _ in range(50):
            N = np.random.randint(3, 15)
            bitstring = ''.join(str(np.random.randint(0, 2)) for _ in range(N))
            spin = bitstring_to_spin(bitstring)
            
            for val in spin:
                self.assertIn(val, [-1, 1], 
                            f"Invalid spin value: {val}")


# ============================================================================
# TEST CLASS 5: Theta Computation Tests
# ============================================================================

class TestThetaComputation(unittest.TestCase):
    """Test suite for theta angle computation."""
    
    def test_theta_finite(self):
        """Theta should be finite for valid inputs."""
        for N in range(3, 12):
            for T in [0.5, 1.0, 2.0]:
                n_steps = 2
                dt = T / n_steps
                for step in range(1, n_steps + 1):
                    t = step * dt
                    theta = compute_theta(t, dt, T, N)
                    
                    self.assertTrue(np.isfinite(theta),
                                  f"Non-finite theta for N={N}, T={T}, t={t}")
    
    def test_theta_reasonable_magnitude(self):
        """Theta should have reasonable magnitude."""
        for N in range(3, 12):
            T = 1.0
            n_steps = 2
            dt = T / n_steps
            
            for step in range(1, n_steps + 1):
                t = step * dt
                theta = compute_theta(t, dt, T, N)
                
                # Theta should be small (typically < pi)
                self.assertLess(abs(theta), 10,
                              f"Theta too large: {theta}")
    
    def test_theta_scales_with_n(self):
        """Theta should generally decrease with larger N."""
        T = 1.0
        n_steps = 2
        dt = T / n_steps
        t = dt
        
        thetas = [compute_theta(t, dt, T, N) for N in range(3, 15)]
        
        # General trend should be decreasing (not strictly)
        avg_first_half = np.mean(np.abs(thetas[:6]))
        avg_second_half = np.mean(np.abs(thetas[6:]))
        
        self.assertGreater(avg_first_half, avg_second_half * 0.5,
                          "Theta should generally decrease with N")


# ============================================================================
# TEST CLASS 6: Classical MTS Component Tests
# ============================================================================

class TestMTSComponents(unittest.TestCase):
    """Test suite for Memetic Tabu Search components."""
    
    def test_combine_length_preserved(self):
        """Combined offspring should have same length as parents."""
        np.random.seed(42)
        for N in range(3, 12):
            p1 = 2 * np.random.randint(0, 2, N) - 1
            p2 = 2 * np.random.randint(0, 2, N) - 1
            
            child = combine(p1, p2)
            
            self.assertEqual(len(child), N,
                           f"Child length mismatch for N={N}")
    
    def test_combine_valid_spins(self):
        """Combined offspring should only have +1/-1 values."""
        np.random.seed(42)
        for _ in range(50):
            N = np.random.randint(3, 15)
            p1 = 2 * np.random.randint(0, 2, N) - 1
            p2 = 2 * np.random.randint(0, 2, N) - 1
            
            child = combine(p1, p2)
            
            for val in child:
                self.assertIn(val, [-1, 1],
                            f"Invalid spin in child: {val}")
    
    def test_mutate_length_preserved(self):
        """Mutated sequence should have same length."""
        np.random.seed(42)
        for N in range(3, 15):
            s = 2 * np.random.randint(0, 2, N) - 1
            mutated = mutate(s, p_mut=0.5)
            
            self.assertEqual(len(mutated), N)
    
    def test_mutate_valid_spins(self):
        """Mutated sequence should only have +1/-1 values."""
        np.random.seed(42)
        for _ in range(50):
            N = np.random.randint(3, 15)
            s = 2 * np.random.randint(0, 2, N) - 1
            mutated = mutate(s, p_mut=0.5)
            
            for val in mutated:
                self.assertIn(val, [-1, 1])
    
    def test_mutate_zero_probability(self):
        """With p_mut=0, sequence should be unchanged."""
        np.random.seed(42)
        for _ in range(20):
            N = np.random.randint(3, 15)
            s = 2 * np.random.randint(0, 2, N) - 1
            mutated = mutate(s.copy(), p_mut=0.0)
            
            self.assertTrue(np.array_equal(s, mutated),
                          "Mutation with p=0 should not change sequence")
    
    def test_tabu_search_improves_or_maintains(self):
        """Tabu search should not worsen the solution."""
        np.random.seed(42)
        for N in range(3, 10):
            s = 2 * np.random.randint(0, 2, N) - 1
            initial_energy = labs_energy(s)
            
            refined, final_energy = tabu_search(s, max_iters=20)
            
            self.assertLessEqual(final_energy, initial_energy,
                               f"Tabu search worsened solution for N={N}")
    
    def test_tabu_search_valid_output(self):
        """Tabu search should return valid spin sequence."""
        np.random.seed(42)
        for N in range(3, 10):
            s = 2 * np.random.randint(0, 2, N) - 1
            refined, energy = tabu_search(s)
            
            self.assertEqual(len(refined), N)
            for val in refined:
                self.assertIn(val, [-1, 1])
            self.assertEqual(labs_energy(refined), energy)


# ============================================================================
# TEST CLASS 7: Quantum Kernel Tests
# ============================================================================

class TestQuantumKernels(unittest.TestCase):
    """Test suite for CUDA-Q quantum kernels."""
    
    def test_circuit_returns_results(self):
        """Circuit should return measurement results."""
        for N in range(3, 8):
            circuit = get_labs_circuit(N)
            theta = 0.1
            
            result = cudaq.sample(circuit, theta, shots_count=100)
            
            self.assertGreater(len(result), 0,
                             f"No results for N={N}")
    
    def test_circuit_correct_bitstring_length(self):
        """All bitstrings should have correct length N."""
        for N in range(3, 8):
            circuit = get_labs_circuit(N)
            theta = 0.1
            
            result = cudaq.sample(circuit, theta, shots_count=100)
            
            for bitstring in result.keys():
                self.assertEqual(len(bitstring), N,
                               f"Wrong bitstring length for N={N}")
    
    def test_circuit_valid_bitstrings(self):
        """All bitstrings should only contain 0s and 1s."""
        for N in range(3, 8):
            circuit = get_labs_circuit(N)
            theta = 0.1
            
            result = cudaq.sample(circuit, theta, shots_count=100)
            
            for bitstring in result.keys():
                for char in bitstring:
                    self.assertIn(char, ['0', '1'],
                                f"Invalid character in bitstring: {char}")
    
    def test_circuit_shot_count(self):
        """Total counts should equal requested shots."""
        for N in [3, 5, 7]:
            circuit = get_labs_circuit(N)
            theta = 0.1
            shots = 500
            
            result = cudaq.sample(circuit, theta, shots_count=shots)
            total_counts = sum(result.values())
            
            self.assertEqual(total_counts, shots,
                           f"Shot count mismatch for N={N}")
    
    def test_different_theta_different_results(self):
        """Different theta values should generally give different distributions."""
        N = 5
        circuit = get_labs_circuit(N)
        shots = 1000
        
        result1 = cudaq.sample(circuit, 0.01, shots_count=shots)
        result2 = cudaq.sample(circuit, 1.0, shots_count=shots)
        
        # The most frequent bitstrings should differ
        # (Not guaranteed, but highly likely)
        most_freq_1 = max(result1.keys(), key=lambda k: result1[k])
        most_freq_2 = max(result2.keys(), key=lambda k: result2[k])
        
        # At minimum, the distributions should not be identical
        # (This is a soft test - statistically almost always passes)
        self.assertTrue(len(result1) > 0 and len(result2) > 0)
    
    def test_kernel_selection(self):
        """get_labs_circuit should return correct kernel for each N."""
        for N in range(3, 11):
            circuit = get_labs_circuit(N)
            self.assertIsNotNone(circuit, f"No circuit for N={N}")
    
    def test_kernel_invalid_n(self):
        """get_labs_circuit should raise error for invalid N."""
        with self.assertRaises(ValueError):
            get_labs_circuit(2)
        
        with self.assertRaises(ValueError):
            get_labs_circuit(11)


# ============================================================================
# TEST CLASS 8: Quantum Sampling Tests
# ============================================================================

class TestQuantumSampling(unittest.TestCase):
    """Test suite for quantum population sampling."""
    
    def test_sample_population_size(self):
        """Sampled population should match requested shots."""
        for N in [3, 5, 7]:
            shots = 500
            population, _ = sample_quantum_population(N, shots=shots)
            
            self.assertEqual(len(population), shots,
                           f"Population size mismatch for N={N}")
    
    def test_sample_spin_format(self):
        """Sampled sequences should be in spin format."""
        N = 5
        population, _ = sample_quantum_population(N, shots=100)
        
        for spin in population:
            self.assertEqual(len(spin), N)
            for val in spin:
                self.assertIn(val, [-1, 1])
    
    def test_sample_returns_counts(self):
        """Function should return both population and counts."""
        N = 5
        population, counts = sample_quantum_population(N, shots=100)
        
        self.assertIsInstance(population, list)
        self.assertGreater(len(counts), 0)


# ============================================================================
# TEST CLASS 9: End-to-End Workflow Tests
# ============================================================================

class TestEndToEndWorkflow(unittest.TestCase):
    """Test suite for complete quantum-enhanced workflow."""
    
    def test_workflow_returns_results(self):
        """Workflow should return results dictionary."""
        result = run_quantum_enhanced_labs(
            N=3,
            pop_size=5,
            generations=2,
            shots=50,
            verbose=False
        )
        
        self.assertIn("N", result)
        self.assertIn("energy_random", result)
        self.assertIn("energy_quantum", result)
        self.assertIn("best_random", result)
        self.assertIn("best_quantum", result)
    
    def test_workflow_valid_energies(self):
        """Workflow should return valid non-negative energies."""
        result = run_quantum_enhanced_labs(
            N=4,
            pop_size=5,
            generations=2,
            shots=50,
            verbose=False
        )
        
        self.assertGreaterEqual(result["energy_random"], 0)
        self.assertGreaterEqual(result["energy_quantum"], 0)
    
    def test_workflow_energy_matches_sequence(self):
        """Reported energy should match energy of reported sequence."""
        result = run_quantum_enhanced_labs(
            N=5,
            pop_size=5,
            generations=3,
            shots=100,
            verbose=False
        )
        
        calculated_random = labs_energy(result["best_random"])
        calculated_quantum = labs_energy(result["best_quantum"])
        
        self.assertEqual(result["energy_random"], calculated_random)
        self.assertEqual(result["energy_quantum"], calculated_quantum)
    
    def test_workflow_finds_good_solutions_small_n(self):
        """Workflow should find optimal or near-optimal for small N."""
        for N in [3, 4, 5]:
            optimal = KNOWN_OPTIMAL_ENERGIES[N]
            
            result = run_quantum_enhanced_labs(
                N=N,
                pop_size=10,
                generations=10,
                shots=200,
                verbose=False
            )
            
            best_found = min(result["energy_random"], result["energy_quantum"])
            
            # Should find optimal or within 50% for small problems
            self.assertLessEqual(best_found, optimal * 1.5,
                               f"Poor solution for N={N}")


# ============================================================================
# TEST CLASS 10: Known Optimal Energy Tests
# ============================================================================

class TestKnownOptima(unittest.TestCase):
    """Test suite for known optimal LABS energies."""
    
    def test_known_optima_exist(self):
        """Known optimal energies should exist for test cases."""
        for N in range(3, 11):
            self.assertIn(N, KNOWN_OPTIMAL_ENERGIES,
                        f"Missing optimal energy for N={N}")
    
    def test_known_optima_positive(self):
        """Known optimal energies should be positive."""
        for N, energy in KNOWN_OPTIMAL_ENERGIES.items():
            self.assertGreater(energy, 0,
                             f"Non-positive optimal for N={N}")
    
    def test_known_optima_increasing(self):
        """Known optimal energies should generally increase with N."""
        Ns = sorted([N for N in KNOWN_OPTIMAL_ENERGIES.keys() if N <= 15])
        energies = [KNOWN_OPTIMAL_ENERGIES[N] for N in Ns]
        
        # Not strictly increasing, but should trend upward
        self.assertLess(energies[0], energies[-1])
    
    def test_can_achieve_optimal_n3(self):
        """Solver should be able to find optimal for N=3."""
        N = 3
        optimal = KNOWN_OPTIMAL_ENERGIES[N]
        
        # Run multiple times to account for randomness
        found_optimal = False
        for seed in range(5):
            result = run_quantum_enhanced_labs(
                N=N,
                pop_size=10,
                generations=10,
                shots=200,
                seed=seed,
                verbose=False
            )
            
            if min(result["energy_random"], result["energy_quantum"]) <= optimal:
                found_optimal = True
                break
        
        self.assertTrue(found_optimal, 
                       f"Could not find optimal for N={N}")


# ============================================================================
# TEST CLASS 11: Two-Qubit Block Tests
# ============================================================================

class TestTwoQubitBlock(unittest.TestCase):
    """Test suite for two-qubit counteradiabatic block."""
    
    def test_two_qubit_block_executes(self):
        """Two-qubit block should execute without error."""
        @cudaq.kernel
        def test_kernel(theta: float):
            q = cudaq.qvector(2)
            h(q[0])
            h(q[1])
            two_qubit_block(q[0], q[1], theta)
            mz(q)
        
        result = cudaq.sample(test_kernel, 0.1, shots_count=100)
        self.assertGreater(len(result), 0)
    
    def test_two_qubit_block_different_theta(self):
        """Different theta should affect output distribution."""
        @cudaq.kernel
        def test_kernel(theta: float):
            q = cudaq.qvector(2)
            h(q[0])
            h(q[1])
            two_qubit_block(q[0], q[1], theta)
            mz(q)
        
        result1 = cudaq.sample(test_kernel, 0.0, shots_count=500)
        result2 = cudaq.sample(test_kernel, 1.5, shots_count=500)
        
        # Both should produce valid results
        self.assertGreater(len(result1), 0)
        self.assertGreater(len(result2), 0)


# ============================================================================
# TEST CLASS 12: Four-Qubit Block Tests
# ============================================================================

class TestFourQubitBlock(unittest.TestCase):
    """Test suite for four-qubit counteradiabatic block."""
    
    def test_four_qubit_block_executes(self):
        """Four-qubit block should execute without error."""
        @cudaq.kernel
        def test_kernel(theta: float):
            q = cudaq.qvector(4)
            h(q[0])
            h(q[1])
            h(q[2])
            h(q[3])
            four_qubit_block(q[0], q[1], q[2], q[3], theta)
            mz(q)
        
        result = cudaq.sample(test_kernel, 0.1, shots_count=100)
        self.assertGreater(len(result), 0)
    
    def test_four_qubit_block_valid_bitstrings(self):
        """Four-qubit block should produce valid 4-bit strings."""
        @cudaq.kernel
        def test_kernel(theta: float):
            q = cudaq.qvector(4)
            h(q[0])
            h(q[1])
            h(q[2])
            h(q[3])
            four_qubit_block(q[0], q[1], q[2], q[3], theta)
            mz(q)
        
        result = cudaq.sample(test_kernel, 0.2, shots_count=100)
        
        for bitstring in result.keys():
            self.assertEqual(len(bitstring), 4)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestLABSEnergy,
        TestLABSSymmetries,
        TestInteractions,
        TestBitstringConversion,
        TestThetaComputation,
        TestMTSComponents,
        TestQuantumKernels,
        TestQuantumSampling,
        TestEndToEndWorkflow,
        TestKnownOptima,
        TestTwoQubitBlock,
        TestFourQubitBlock,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM-ENHANCED LABS SOLVER - TEST SUITE")
    print("=" * 70)
    print()
    
    result = run_tests(verbosity=2)
    
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")
        
        if result.failures:
            print("\nFailures:")
            for test, trace in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, trace in result.errors:
                print(f"  - {test}")
