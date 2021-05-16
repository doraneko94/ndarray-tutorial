use ndarray::*;

trait SkewArr<D: RemoveAxis> {
    fn skew_axis(&self, axis: Axis) -> Option<Array<f64, D::Smaller>>;
}

impl<D: RemoveAxis> SkewArr<D> for Array<f64, D> {
    fn skew_axis(&self, axis: Axis) -> Option<Array<f64, D::Smaller>> {
        let index = axis.index();
        if self.ndim() < index {
            return None;
        }

        let n = self.shape()[index] as f64;
        if n <= 2.0 {
            return None;
        }

        let mut ans = Array::zeros(self.dim().clone()).remove_axis(axis);
        let mu = self.mean_axis(axis).unwrap();
        let sigma = self.std_axis(axis, 1.0);

        for self_sub in self.axis_iter(axis) {
            Zip::from(&mut ans)
                .and(self_sub)
                .and(&mu)
                .and(&sigma)
                .for_each(|a, &b, &c, &d| {
                    *a += ((b - c) / d).powi(3);
                })
        }

        let c = n / ((n - 1.0) * (n - 2.0));
        Zip::from(&mut ans)
            .for_each(|a| *a *= c);

        Some(ans)
    }
}

fn main() {
    let a = arr2(&[
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0],
    ]);

    println!("a.skew_axis(Axis(0)).unwrap() = ");
    println!("{:?}", a.skew_axis(Axis(0)).unwrap());
    println!("********************************");
    println!("a.skew_axis(Axis(1)).unwrap() = ");
    println!("{:?}", a.skew_axis(Axis(1)).unwrap());
}