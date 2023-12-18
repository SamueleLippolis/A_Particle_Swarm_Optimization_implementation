/*-
 * ========================LICENSE_START=================================
 * jgea-core
 * %%
 * Copyright (C) 2018 - 2023 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */
package io.github.ericmedvet.jgea.core.solver;

import static io.github.ericmedvet.jgea.core.util.VectorUtils.*;

import io.github.ericmedvet.jgea.core.Factory;
import io.github.ericmedvet.jgea.core.order.PartiallyOrderedCollection;
import io.github.ericmedvet.jgea.core.problem.TotalOrderQualityBasedProblem;
import io.github.ericmedvet.jgea.core.util.Progress;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.random.RandomGenerator;

// PSO class
//    S is the space of the solutions (then, then space of the particles)
//    Q is the space of the fitness function (in pso, in general, it is R)
// The class extends AbstractPopulationBasedIterativeSolver
//    It is at least an Abstract population based iterative solver
//    Indeed, the method involves a population of solutions that is updated iteration per iteration.
// It uses "PSOIndividual"
//    A PSO individual stores the position, velocity, own best known position
//    The previous implemented Individual interface does not allow to store velocity and own best known position
//    So, we built the interface called PSO individual
public class ParticleSwarmOptimization<S, Q>
    extends AbstractPopulationBasedIterativeSolver<
        ListPopulationState<ParticleSwarmOptimization.PSOIndividual<S, Q>, List<Double>, S, Q>,
        TotalOrderQualityBasedProblem<S, Q>,
        ParticleSwarmOptimization.PSOIndividual<S, Q>,
        List<Double>,
        S,
        Q> {

  // PSOIndividual
  //    It is at least an Individual, so it inherits the fields: genotype, phenotype and fitness
  //    Moreover, we add the velocity and the best know position found so far by the individual
  // Parameters
  //    The PSOIndividual interface is parametrized by the two parameters S (phenotype) and Q (fitness)
  public interface PSOIndividual<S, Q> extends Individual<List<Double>, S, Q> {

    // 3 further methods
    //    They allow to manage the characterizing features of a PSO individual
    List<Double> velocity();

    List<Double> bestKnownPosition();

    Q bestKnownQuality();

    // A default implementation for the position method
    //    Indeed, in our code, the pso algorithm has POSITION = GENOTYPE
    //    However, the method can be overridden
    default List<Double> position() {
      return genotype();
    }

    // This method allow to create instance of the PSOIndividual
    //    It takes the parameters representing the state of the individual
    //    It returns an instance of the class HardIndividual (which implement PSO individual)
    static <S1, Q1> PSOIndividual<S1, Q1> of(
        List<Double> genotype,
        List<Double> velocity,
        List<Double> bestKnownPosition,
        Q1 bestKnownQuality,
        S1 solution,
        Q1 quality,
        long genotypeBirthIteration,
        long qualityMappingIteration) {
      record HardIndividual<S1, Q1>(
          List<Double> genotype,
          List<Double> velocity,
          List<Double> bestKnownPosition,
          Q1 bestKnownQuality,
          S1 solution,
          Q1 quality,
          long genotypeBirthIteration,
          long qualityMappingIteration)
          implements PSOIndividual<S1, Q1> {}
      return new HardIndividual<>(
          genotype,
          velocity,
          bestKnownPosition,
          bestKnownQuality,
          solution,
          quality,
          genotypeBirthIteration,
          qualityMappingIteration);
    }
  }

  // Parameters of PSO
  //    w = inertia parameter (it determines the tendency to maintain the same velocity)
  //      If |w| < 1 we can say it is a dumping factor
  //    phiParticle = tendency to follow the fittest position found so far by the particle
  //    phiGlobal = tendency to follow the fittest position found so far by the swarm
  private final int populationSize;
  private final double w;
  private final double phiParticle;
  private final double phiGlobal;



  // State is a record and represents the state of the algorithm
  // It includes information as
  //    number of fitness evaluations (it will be the stopping criterion)
  //    the population
  //    the individual that has found the fittest position so far (useful for the velocity-update)
  protected record State<S, Q>(
      LocalDateTime startingDateTime,
      long elapsedMillis,
      long nOfIterations,
      Progress progress,
      long nOfBirths,
      long nOfFitnessEvaluations,
      PartiallyOrderedCollection<PSOIndividual<S, Q>> pocPopulation,
      List<PSOIndividual<S, Q>> listPopulation,
      PSOIndividual<S, Q> knownBest)
      implements ListPopulationState<PSOIndividual<S, Q>, List<Double>, S, Q> {

    // It creates a new State instance
    // It is used to update an existing one
    public static <S, Q> State<S, Q> from(
        State<S, Q> state,
        Progress progress,
        int nOfBirths,
        int nOfFitnessEvaluations,
        Collection<PSOIndividual<S, Q>> listPopulation,
        PSOIndividual<S, Q> knownBest,
        Comparator<? super PSOIndividual<S, Q>> comparator) {
      return new State<>(
          state.startingDateTime,
          ChronoUnit.MILLIS.between(state.startingDateTime, LocalDateTime.now()),
          state.nOfIterations() + 1,
          progress,
          state.nOfBirths() + nOfBirths,
          state.nOfFitnessEvaluations() + nOfFitnessEvaluations,
          PartiallyOrderedCollection.from(listPopulation, comparator),
          listPopulation.stream().sorted(comparator).toList(),
          knownBest);
    }

    // It is a method to create an instance of state
    // It is used to build an instance of State for the first time
    //   Given a collection of Individuals and a comparator
    //   It initializes the state with 0 elapsed time, 0 number of iterations and so on
    public static <S, Q> State<S, Q> from(
        Collection<PSOIndividual<S, Q>> listPopulation, Comparator<? super PSOIndividual<S, Q>> comparator) {
      List<PSOIndividual<S, Q>> list =
          listPopulation.stream().sorted(comparator).toList();
      return new State<>(
          LocalDateTime.now(),
          0,
          0,
          Progress.NA,
          listPopulation.size(),
          listPopulation.size(),
          PartiallyOrderedCollection.from(listPopulation, comparator),
          listPopulation.stream().sorted(comparator).toList(),
          list.get(0));
    }
  }

  // The proper PSO class
  //  solutionMapper: it maps genotype in G = R^p in a solution in S
  //  genotypeFactory: responsible of generating the initial population
  //  stopCondition: a predicate the determines when stop the algorithm
  // Constructor
  //  We call the Constructor from  AbstractPopulationBasedIterativeSolver
  public ParticleSwarmOptimization(
      Function<? super List<Double>, ? extends S> solutionMapper,
      Factory<? extends List<Double>> genotypeFactory,
      Predicate<? super ListPopulationState<PSOIndividual<S, Q>, List<Double>, S, Q>> stopCondition,
      int populationSize,
      double w,
      double phiParticle,
      double phiGlobal) {
    super(solutionMapper, genotypeFactory, stopCondition, false);
    this.populationSize = populationSize;
    this.w = w;
    this.phiParticle = phiParticle;
    this.phiGlobal = phiGlobal;
  }

  // This method is not useful for PSO.
  // However, it must be overridden (since it appears in AbstractPopulationBasedSolver)
  @Override
  protected PSOIndividual<S, Q> newIndividual(
      List<Double> genotype,
      ListPopulationState<PSOIndividual<S, Q>, List<Double>, S, Q> state,
      TotalOrderQualityBasedProblem<S, Q> problem) {
    throw new UnsupportedOperationException("This method should not be called");
  }

  // This method is not useful for PSO.
  // However, it must be overridden (since it appears in AbstractPopulationBasedSolver)
  @Override
  protected PSOIndividual<S, Q> updateIndividual(
      PSOIndividual<S, Q> individual,
      ListPopulationState<PSOIndividual<S, Q>, List<Double>, S, Q> state,
      TotalOrderQualityBasedProblem<S, Q> problem) {
    throw new UnsupportedOperationException("This method should not be called");
  }


  // The init method is used to generate a first population
  //  Thanks to "genotypeFactory", a random population (so the genotypes) is built in R^p in the interval [-1,1]^p
  //  The best position know so far are the positions themselves
  //  The velocity is done by generating random numbers inside an interval (build using the max and min values of the generated positions)
  @Override
  public ListPopulationState<PSOIndividual<S, Q>, List<Double>, S, Q> init(
      TotalOrderQualityBasedProblem<S, Q> problem, RandomGenerator random, ExecutorService executor)
      throws SolverException {
    // init positions
    List<? extends List<Double>> positions = genotypeFactory.build(populationSize, random);
    double min = positions.stream()
        .flatMap(List::stream)
        .mapToDouble(v -> v)
        .min()
        .orElseThrow();
    double max = positions.stream()
        .flatMap(List::stream)
        .mapToDouble(v -> v)
        .max()
        .orElseThrow();
    try {
      Collection<PSOIndividual<S, Q>> individuals = getAll(executor.invokeAll(positions.stream()
          .map(p -> (Callable<PSOIndividual<S, Q>>) () -> {
            S s = solutionMapper.apply(p);
            Q q = problem.qualityFunction().apply(s);
            return PSOIndividual.of(
                p,
                buildList(p.size(), () -> random.nextDouble(-Math.abs(max - min), Math.abs(max - min))),
                p,
                q,
                s,
                q,
                0,
                0);
          })
          .toList()));
      return State.from(individuals, comparator(problem));
    } catch (InterruptedException e) {
      throw new SolverException(e);
    }
  }

  // The update method is responsible for the swarm update in the PSO procedure
  //  Firstly, it retrieves the fittest position found so far by each particle (bestKnown) and the fittest position found by the swarm (globalBestPosition)
  // Inside the try block there is the actual position/velocity/bestKnown update
  // Return an instance of State (with individuals ordered depending on the fitness)
  @Override
  public ListPopulationState<PSOIndividual<S, Q>, List<Double>, S, Q> update(
      TotalOrderQualityBasedProblem<S, Q> problem,
      RandomGenerator random,
      ExecutorService executor,
      ListPopulationState<PSOIndividual<S, Q>, List<Double>, S, Q> state)
      throws SolverException {
    PSOIndividual<S, Q> knownBest = ((State<S, Q>) state).knownBest();
    List<Double> globalBestPosition = knownBest.position();
    try {
      Collection<PSOIndividual<S, Q>> individuals = getAll(executor.invokeAll(((State<S, Q>) state)
          .listPopulation.stream()
              .map(i -> (Callable<PSOIndividual<S, Q>>) () -> {
                double rParticle = random.nextDouble();
                double rGlobal = random.nextDouble();
                List<Double> vVel = mult(i.velocity(), w);
                List<Double> vParticle =
                    mult(diff(i.bestKnownPosition(), i.position()), rParticle * phiParticle);
                List<Double> vGlobal =
                    mult(diff(globalBestPosition, i.position()), rGlobal * phiGlobal);
                List<Double> newVelocity = sum(
                    vVel, sum(vParticle, vGlobal)); // TODO maybe make a sum version with varargs
                List<Double> newPosition = sum(i.position(), newVelocity);
                S newSolution = solutionMapper.apply(newPosition);
                Q newQuality = problem.qualityFunction().apply(newSolution);
                List<Double> newBestKnownPosition = i.bestKnownPosition();
                Q newBestKnownQuality = i.bestKnownQuality();
                if (problem.totalOrderComparator().compare(newQuality, i.quality()) < 0) {
                  newBestKnownPosition = newPosition;
                  newBestKnownQuality = newQuality;
                }
                return PSOIndividual.of(
                    newPosition,
                    newVelocity,
                    newBestKnownPosition,
                    newBestKnownQuality,
                    newSolution,
                    newQuality,
                    state.nOfIterations(),
                    state.nOfIterations());
              })
              .toList()));
      List<PSOIndividual<S, Q>> sortedIndividuals =
          individuals.stream().sorted(comparator(problem)).toList();

      if (comparator(problem).compare(sortedIndividuals.get(0), knownBest) < 0) {
        knownBest = sortedIndividuals.get(0);
      }
      return State.from(
          (State<S, Q>) state,
          progress(state),
          populationSize,
          populationSize,
          sortedIndividuals,
          knownBest,
          comparator(problem));
    } catch (InterruptedException e) {
      throw new SolverException(e);
    }
  }
}
