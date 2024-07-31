use rand;
use rand::Rng;
use std::fmt;
use threadpool::Builder;
use std::sync::mpsc::channel;
use clap::Parser;

type Coordinates = (usize, usize);

#[derive(Clone, PartialEq, Eq, Debug)]
enum CellState {
    Empty,
    Tree,
    Fire,
}

impl fmt::Display for CellState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CellState::Empty => write!(f, "⬛"),
            CellState::Tree => write!(f, "🌳"),
            CellState::Fire => write!(f, "🔥"),
        }
    }
}

struct Forest {
    /// Number of rows in the forest.
    rows: usize,
    /// Number of columns in the forest.
    columns: usize,
    /// A collection of all the cells in the forest. Rather than mess around with Vecs in vecs, we just 
    /// have a single vec and owrk out the index into this for a given set of coordinates.
    cells: Vec<CellState>,
    /// Vec of cells that are currently on fire. This means we only check the cells that could change on 
    /// each iteration, rather than the entire grid. 
    on_fire: Vec<Coordinates>,
}

impl Forest {
    /// Create a new version of the forest with a given density of trees.
    fn new(rows: usize, columns: usize, density: f64) -> Self {
        // To make things easier, we pad the outside of the forest with empty cells, as this means we 
        // don't have to do tedious bounds checking. 
        let num_cells = (rows+2) * (columns+2);
        let cells: Vec<CellState> = vec![CellState::Empty; num_cells];
        let mut forest = Forest { rows, columns, cells, on_fire: Vec::new() };

        // Populate the forest with trees. Because of the padding, we don't consider the first or last column or row. 
        let mut rng = rand::thread_rng();
        for r in 1..=rows {
            for c in 1..=columns {
                let sample: f64 = rng.gen();
                if sample < density {
                    *forest.get_cell_mut((r, c)) = CellState::Tree;
                }
            }
        }

        forest
    }

    /// Get a mutable refernce to a cell. 
    fn get_cell_mut(&mut self, coords: Coordinates) -> &mut CellState {
        let (r, c) = coords;
        &mut self.cells[r * (self.columns + 2) + c as usize]
    }

    /// Get an immutable refernce to a cell. 
    fn get_cell(&self, coords: Coordinates) -> &CellState {
        let (r, c) = coords;
        &self.cells[r * (self.columns + 2) + c as usize]
    }

    /// Get a mutable reference to an entire row. Note that rows are held together in memory, so this is just a slice.
    fn get_row_mut(&mut self, row: usize) -> &mut [CellState] {
        let start = row * (self.columns + 2) + 1;
        &mut self.cells[start..start + self.columns]
    }

    /// Start the fire. (Make up your own Billy Joel joke here).
    fn start_fire(&mut self) {
        let mut on_fire = Vec::new();

        for (ix, cell) in self.get_row_mut(1).iter_mut().enumerate() {
            if *cell == CellState::Tree {
                // Mark each tree cell as being on fire, and add the coordinate to the list of on-fire cells.
                *cell = CellState::Fire;
                on_fire.push((1, ix+1));
            }
        }

        self.on_fire = on_fire;
    }

    /// Perform a single iteration of the fire.
    fn advance(&mut self) {
        let on_fire = std::mem::replace(&mut self.on_fire, Vec::new());

        // Only cells that are currently on fire can spread fire to adjacent cells, so loop over them. 
        for (row, col) in on_fire.iter() {
            *(self.get_cell_mut((*row, *col))) = CellState::Empty;

            // Loop over the cell's neightbours, setting them on fire if they contain a tree.
            for coord in [(*row-1, *col), (*row+1, *col), (*row, *col-1), (*row, *col+1)].iter() {
                let cell = self.get_cell_mut(*coord);
                if *cell == CellState::Tree {
                    *cell = CellState::Fire;
                    self.on_fire.push(*coord);
                }
            }
        }
    }

    /// Check if the fire has percolate. That's the case if it's reached the final row. 
    fn percolated(&self) -> bool {
        self.on_fire.iter().any(|(r, _)| {
            *r == self.rows
        })
    }

    /// Check if the fire is still burning (since the world's been turning... OK I'll stop now)
    fn fire_burning(&self) -> bool {
        !self.on_fire.is_empty()
    }

    // Run a single simulation to completion.
    fn simulate(&mut self) -> bool {
        //let _ = write!(std::io::stdout(), ".");
        //let _ = std::io::stdout().flush();
        self.start_fire();

        while self.fire_burning() {
            self.advance();
            if self.percolated() {
                return true;
            }
        }
        return false;
    }
}

impl fmt::Display for Forest {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for r in 1..=self.rows {
            for c in 1..=self.columns {
                write!(f, "{}", self.get_cell((r, c)))?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

/// Perform n simulations across multiple CPU cores.
fn simulate_n(rows: usize, columns: usize, density: f64, count: usize) -> usize {
    let pool = Builder::new().build();
    let (tx, rx) = channel();

    for _ in 0..count {
        let tx = tx.clone();
        pool.execute(move || {
            let mut forest = Forest::new(rows, columns, density);
            tx.send(forest.simulate()).unwrap();
        });
    }
    drop(tx);

    rx.into_iter().filter(|x| *x).count()
}

/// Program to simulate forest fire percolation
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The number of rows in the forest
    #[arg(short, long)]
    rows: usize,

    /// The number of columns in the forest
    #[arg(short, long)]
    columns: usize,

    /// The maximum number of simulations to run at each step
    #[arg(short, long, default_value_t = 1_000)]
    max_simulations: usize,

    /// The simulation only proceeds to the next step when the estimate of the percolation probability
    /// is far enough above or below 0.5. "Far enough" depends on the error in the estimate, and the 
    /// confident level: z_crit.  
    /// 
    /// We default this to 3.09, which gives a 1 in 1000 chance of being wrong.
    #[arg(short, long, default_value_t = 3.09)]
    z_crit: f64,
}

const MICROBATCH: usize = 100;

fn main() {
    let args = Args::parse();

    println!("density,percolation");

    // Binary chop to find the percolation threshold. Lo and hi define the range we are searching in, and 
    // lo_prob and hi_prob are the percolation probabilites at those points.
    let mut lo = 0f64;
    let mut hi = 1.0f64;
    let mut lo_prob = 0.0f64; 
    let mut hi_prob = 1.0f64; 

    loop {
        // Find the midpoint of the range. 
        //
        // To find this faster, we assume that the percolation probability is linear between lo and hi. 
        // This means that we converge faster if either hi or lo is very close to the threshold (0.5).
        // If the assumption is wrong, the algorithm still works, it may just atke a bit longer to converge.
        let mid = hi - (hi_prob - 0.5)/(hi_prob - lo_prob)*(hi - lo);

        let mut simulation_count = 0;
        let mut percolation_count = 0;
        
        while simulation_count < args.max_simulations {
            percolation_count += simulate_n(args.rows, args.columns, mid, MICROBATCH);
            simulation_count += MICROBATCH;

            // Estimate the percolation probability. 
            let p_hat = percolation_count as f64 / simulation_count as f64;

            // Here's where things get interesting. Because the simulation is stochastic, we p_hat has error
            // bars on it. To do the binary chop we need to know definitively if the midpoint is below or above 
            // the threshold. But if we get this wrong (because of the error) we will end up looking in the
            // wrong range and finding the wrong answer. 
            //
            // To avoid this we calclate the standard deviation of **our estimate of p_hat**. Since the underlying
            // simulation is a binomial distribution, the s.d. is sqrt(np(1-p)), so the s.d. of the estimate of the
            // mean is sqrt(p(1-p)/n). We keep doing simulations until p_hat is far enough away from 0.5. "Far enough"
            // is controlled by the z_crit argument.
            let sigma = f64::sqrt(p_hat * (1.0 - p_hat) / simulation_count as f64);

            //println!("  p_hat={p_hat}, sigma={sigma}, count={simulation_count}");
            
            // If we're far enough away from 0.5, we're done with this iteration.
            if p_hat - args.z_crit * sigma > 0.5 {
                hi = mid;
                hi_prob = p_hat;
                break;
            } else if p_hat + args.z_crit * sigma < 0.5 {
                lo = mid;
                lo_prob = p_hat;
                break;
            }
        }
        println!("{},{}", mid, (percolation_count as f64) / (simulation_count as f64));

        if simulation_count >= args.max_simulations {
            // We've done too many simulations on this iteration, and haven't been able to progress, so stop here. 
            // Output our estimate of the density threshold, and the error. 
            let error = f64::min(hi-mid, mid-lo);
            let precision = (1.0 / error).log(10.0).ceil() as usize;
            println!("Threshold={0:.1$}±{2:.3$}", mid, &precision, error, &(precision+1));
            break;
        }
    }
}
