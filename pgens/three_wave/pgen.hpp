#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  struct FMSTag {};
  struct AWTag {};

  // struct to compute the polarizataion of a single wave mode of a given type (FMS or AW)
  template <Dimension D>
  struct WaveEntry {
    enum class Type { FMS, AW };
 
    Type type;
    real_t k[3];
    real_t amplitude;
    real_t phase;

    Inline auto norm() const -> real_t {
      auto k_norm = ZERO;
      for (int d = 0; d < static_cast<int>(D); ++d) k_norm += k[d] * k[d];
      return math::sqrt(k_norm);
    }

    Inline auto kperp_hat(real_t out[3]) const -> void {
      real_t k_norm = norm();
      out[0] = k[0] / k_norm;
      out[1] = k[1] / k_norm;
      out[2] = ZERO;
    }

    Inline auto phase_at(const coord_t<D>& x_Ph) const -> real_t {
      auto k_dot_r = ZERO;
      for (int d = 0; d < static_cast<int>(D); ++d) k_dot_r += k[d] * x_Ph[d];
      return k_dot_r + phase;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      real_t kph[3];
      kperp_hat(kph);
      real_t cp = math::cos(phase_at(x_Ph));
      if (type == Type::FMS) {
        return amplitude * cp * kph[1];
      } else {
        return amplitude * cp * kph[0];
      }
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      real_t kph[3];
      kperp_hat(kph);
      real_t cp = math::cos(phase_at(x_Ph));
      if (type == Type::FMS) {
        return -amplitude * cp * kph[0];
      } else {
        return amplitude * cp * kph[1];
      }
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      real_t kph[3];
      kperp_hat(kph);
      real_t cp    = math::cos(phase_at(x_Ph));
      real_t kn    = norm();
      if (type == Type::FMS) {
        real_t cos_theta = k[2] / kn;
        return amplitude * cp * kph[0] * cos_theta;
      } else {
        real_t sign_kz = (k[2] >= ZERO) ? ONE : -ONE;
        return -amplitude * cp * kph[1] * sign_kz;
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      real_t kph[3];
      kperp_hat(kph);
      real_t cp    = math::cos(phase_at(x_Ph));
      real_t kn    = norm();
      if (type == Type::FMS) {
        real_t cos_theta = k[2] / kn;
        return amplitude * cp * kph[1] * cos_theta;
      } else {
        real_t sign_kz = (k[2] >= ZERO) ? ONE : -ONE;
        return amplitude * cp * kph[0] * sign_kz;
      }
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      real_t cp = math::cos(phase_at(x_Ph));
      real_t kn = norm();
      if (type == Type::FMS) {
        real_t cos_theta = k[2] / kn;
        real_t sin_theta = math::sqrt(ONE - cos_theta * cos_theta);
        return -amplitude * cp * sin_theta;
      } else {
        return ZERO;
      }
    }
  }; // struct WaveEntry

  template <Dimension D, int N>
  struct ExternalCurrent {
    WaveEntry<D> waves[N];

    ExternalCurrent() = default;

    explicit ExternalCurrent(const WaveEntry<D> (&w)[N]) {
      for (int i = 0; i < N; ++i) waves[i] = w[i];
    }

    Inline auto jx1(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto jx2(const coord_t<D>&) const -> real_t { return ZERO; }

    Inline auto jx3(const coord_t<D>& x_Ph) const -> real_t {
      real_t jx3_tot = ZERO;
      for (int i = 0; i < N; ++i) {
        if (waves[i].type == WaveEntry<D>::Type::AW) {
          real_t k_dot_r   = waves[i].k[0] * x_Ph[0] +
                             waves[i].k[1] * x_Ph[1] +
                             waves[i].k[2] * x_Ph[2];
          real_t cos_phase = math::cos(k_dot_r + waves[i].phase);
          real_t k_perp    = math::sqrt(waves[i].k[0] * waves[i].k[0] +
                                        waves[i].k[1] * waves[i].k[1]);
          real_t sign_kz   = (waves[i].k[2] >= ZERO) ? ONE : -ONE;
          jx3_tot         += k_perp * waves[i].amplitude * cos_phase * sign_kz;
        }
      }
      return jx3_tot;
    }
  }; // struct ExternalCurrent


  template <Dimension D, int N>
  struct InitFields {
    WaveEntry<D> waves[N];

    InitFields() = default;

    explicit InitFields(const WaveEntry<D> (&w)[N]) {
      for (int i = 0; i < N; ++i) waves[i] = w[i];
    }

    Inline auto ex1(const coord_t<D>& x) const -> real_t {
      real_t val = ZERO;
      for (int i = 0; i < N; ++i) val += waves[i].ex1(x);
      return val;
    }
    Inline auto ex2(const coord_t<D>& x) const -> real_t {
      real_t val = ZERO;
      for (int i = 0; i < N; ++i) val += waves[i].ex2(x);
      return val;
    }
    Inline auto ex3(const coord_t<D>& x) const -> real_t {
      real_t val = ZERO;
      for (int i = 0; i < N; ++i) val += waves[i].ex3(x);
      return val;
    }
    Inline auto bx1(const coord_t<D>& x) const -> real_t {
      real_t val = ZERO;
      for (int i = 0; i < N; ++i) val += waves[i].bx1(x);
      return val;
    }
    Inline auto bx2(const coord_t<D>& x) const -> real_t {
      real_t val = ZERO;
      for (int i = 0; i < N; ++i) val += waves[i].bx2(x);
      return val;
    }
    Inline auto bx3(const coord_t<D>& x) const -> real_t {
      real_t val = ZERO;
      for (int i = 0; i < N; ++i) val += waves[i].bx3(x);
      return val;
    }
  }; // struct InitFields

  // Store wave entries from the input file
  template <Dimension D, int N>
  auto buildWaveEntries(const SimulationParams& p, WaveEntry<D> (&out)[N]) -> void {
    auto wave_types        = p.template get<std::vector<int>>("setup.wave_type");
    auto wave_kxs          = p.template get<std::vector<real_t>>("setup.wave_kx");
    auto wave_kys          = p.template get<std::vector<real_t>>("setup.wave_ky");
    auto wave_kzs          = p.template get<std::vector<real_t>>("setup.wave_kz");
    auto wave_amps         = p.template get<std::vector<real_t>>("setup.wave_amp");
    auto wave_phases       = p.template get<std::vector<real_t>>("setup.wave_phase");

    for (int i = 0; i < N; ++i) {
      out[i].type      = (wave_types[i] == 0) ? WaveEntry<D>::Type::FMS
                                               : WaveEntry<D>::Type::AW;
      out[i].k[0]      = constant::TWO_PI * wave_kxs[i];
      out[i].k[1]      = constant::TWO_PI * wave_kys[i];
      out[i].k[2]      = constant::TWO_PI * wave_kzs[i];
      out[i].phase     = constant::TWO_PI * wave_phases[i];
      out[i].amplitude = wave_amps[i];
    }
  }

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_3D> {}
    };

    const SimulationParams& params;
    Metadomain<S, M>&       metadomain;

    static constexpr int N_WAVES = 3;

    InitFields<D, N_WAVES>      init_flds;
    ExternalCurrent<D, N_WAVES> ext_current;

  PGen(const SimulationParams& p, Metadomain<S, M>& m)
    : params { p }
    , metadomain { m }
    , init_flds {}
    , ext_current {} {

    // Build once, use for both
    WaveEntry<D> entries[N_WAVES];
    buildWaveEntries<D, N_WAVES>(p, entries);
    init_flds   = InitFields<D, N_WAVES>(entries);
    ext_current = ExternalCurrent<D, N_WAVES>(entries);
  }

  inline void InitPrtls(Domain<S, M>& local_domain) {
    const auto temperature = params.template get<real_t>("setup.temperature", 0.001);
    const auto energy_dist = arch::Maxwellian<S, M>(
                                local_domain.mesh.metric, 
                                local_domain.random_pool(), 
                                temperature);
  arch::InjectUniform<S, M, decltype(energy_dist), decltype(energy_dist)>(
                        params,
                        local_domain,
                        { 1, 2 },
                        { energy_dist, energy_dist },
                        1.0); 
    }

  }; // struct PGen

} // namespace user

#endif
