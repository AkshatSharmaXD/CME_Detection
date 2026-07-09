import { useState, useEffect } from 'react'
import { ArrowLeft, Satellite, AlertTriangle, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { satelliteApi } from '../lib/api'

interface Satellite {
  norad_id: number
  name: string
  object_type?: string
}

interface SatelliteDetails {
  norad_id: number
  name: string
  latitude: number
  longitude: number
  altitude_km?: number
  velocity_km_s?: number
  inclination?: number
  eccentricity?: number
  period_minutes?: number
}

interface CMEPrediction {
  success: boolean
  satellite: SatelliteDetails
  noaa_match?: {
    latitude: number
    longitude: number
    match_distance_degrees: number
    timestamp: string
  }
  wind_parameters?: {
    speed_km_s: number
    density_particles_cm3: number
    temperature_k: number
    bz_gsm_nt: number
    bt_nt: number
  }
  cme_analysis?: {
    probability: number
    occurring: boolean
    risk_level: string
    threshold_used: number
    scores: {
      velocity_score: number
      density_score: number
      temperature_score: number
      bz_score: number
    }
    thresholds: {
      velocity_threshold_km_s: number
      density_threshold_particles_cm3: number
      temperature_threshold_k: number
      bz_threshold_nt: number
    }
  }
  error?: string
}

export default function FieldDataPrediction() {
  const [satellites, setSatellites] = useState<Satellite[]>([])
  const [selectedNoradId, setSelectedNoradId] = useState<number | null>(null)
  const [satelliteDetails, setSatelliteDetails] = useState<SatelliteDetails | null>(null)
  const [cmePrediction, setCmePrediction] = useState<CMEPrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingSatellites, setLoadingSatellites] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [threshold, setThreshold] = useState(0.5)

  // Fetch satellites list on component mount
  useEffect(() => {
    fetchSatellites()
  }, [])

  const fetchSatellites = async () => {
    setLoadingSatellites(true)
    setError(null)
    try {
      const response = await fetch(satelliteApi.list(), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      
      if (data.success) {
        setSatellites(data.satellites || [])
        if (data.satellites && data.satellites.length === 0) {
          setError('No satellites found. The API might be empty or the external service is unavailable.')
        }
      } else {
        setError(`Failed to fetch satellites: ${data.error || data.detail || 'Unknown error'}. Make sure backend is running on port 8002`)
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(`Error fetching satellites: ${errorMessage}. Make sure backend is running on port 8002`)
      console.error('Satellite fetch error:', err)
    } finally {
      setLoadingSatellites(false)
    }
  }

  const handleSatelliteSelect = async (noradId: number) => {
    setSelectedNoradId(noradId)
    setLoading(true)
    setError(null)
    setSatelliteDetails(null)
    setCmePrediction(null)

    try {
      // Fetch satellite details
      const detailsResponse = await fetch(satelliteApi.details(noradId))
      const detailsData = await detailsResponse.json()
      
      if (detailsData.success) {
        setSatelliteDetails(detailsData.satellite)
      } else {
        setError('Failed to fetch satellite details')
        return
      }

      // Fetch CME prediction
      const predictionResponse = await fetch(
        satelliteApi.cmePrediction(noradId, threshold)
      )
      const predictionData = await predictionResponse.json()
      setCmePrediction(predictionData)
      
    } catch (err) {
      setError(`Error fetching data: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const handleRefreshPrediction = async () => {
    if (!selectedNoradId) return
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(
        satelliteApi.cmePrediction(selectedNoradId, threshold)
      )
      const data = await response.json()
      setCmePrediction(data)
    } catch (err) {
      setError(`Error refreshing prediction: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'HIGH':
        return 'text-red-400 bg-red-400/20 border-red-400/50'
      case 'MEDIUM':
        return 'text-yellow-400 bg-yellow-400/20 border-yellow-400/50'
      case 'LOW':
        return 'text-green-400 bg-green-400/20 border-green-400/50'
      default:
        return 'text-gray-400 bg-gray-400/20 border-gray-400/50'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 flex items-center gap-4">
          <div className="p-2">
            <ArrowLeft className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Field Data Future Prediction</h1>
            <p className="text-gray-300">Satellite-based CME occurrence prediction using NOAA wind data</p>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Satellite Selection */}
          <div className="lg:col-span-1 space-y-6">
            {/* Satellite Dropdown */}
            <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <Satellite className="w-5 h-5" />
                Select Satellite
              </h2>
              
              {loadingSatellites ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 text-white animate-spin" />
                </div>
              ) : (
                <select
                  value={selectedNoradId || ''}
                  onChange={(e) => handleSatelliteSelect(Number(e.target.value))}
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">-- Select a satellite --</option>
                  {satellites.map((sat) => (
                    <option key={sat.norad_id} value={sat.norad_id} className="bg-slate-800">
                      {sat.name} (NORAD: {sat.norad_id})
                    </option>
                  ))}
                </select>
              )}

              {/* Threshold Input */}
              <div className="mt-4">
                <label className="block text-sm text-gray-300 mb-2">
                  CME Probability Threshold: {threshold}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>0.0</span>
                  <span>0.5</span>
                  <span>1.0</span>
                </div>
              </div>

              {selectedNoradId && (
                <button
                  onClick={handleRefreshPrediction}
                  disabled={loading}
                  className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Refreshing...
                    </>
                  ) : (
                    'Refresh Prediction'
                  )}
                </button>
              )}
            </div>

            {/* Satellite Details */}
            {satelliteDetails && (
              <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
                <h2 className="text-xl font-semibold text-white mb-4">Satellite Details</h2>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Name:</span>
                    <span className="text-white font-medium">{satelliteDetails.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">NORAD ID:</span>
                    <span className="text-white font-medium">{satelliteDetails.norad_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Latitude:</span>
                    <span className="text-white font-medium">{satelliteDetails.latitude?.toFixed(4)}°</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Longitude:</span>
                    <span className="text-white font-medium">{satelliteDetails.longitude?.toFixed(4)}°</span>
                  </div>
                  {satelliteDetails.altitude_km && (
                    <div className="flex justify-between">
                      <span className="text-gray-300">Altitude:</span>
                      <span className="text-white font-medium">{satelliteDetails.altitude_km.toFixed(2)} km</span>
                    </div>
                  )}
                  {satelliteDetails.velocity_km_s && (
                    <div className="flex justify-between">
                      <span className="text-gray-300">Velocity:</span>
                      <span className="text-white font-medium">{satelliteDetails.velocity_km_s.toFixed(2)} km/s</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - CME Prediction Results */}
          <div className="lg:col-span-2 space-y-6">
            {loading && !cmePrediction && (
              <div className="bg-white/10 backdrop-blur-lg rounded-lg p-12 border border-white/20 flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-white animate-spin" />
              </div>
            )}

            {cmePrediction && (
              <>
                {/* CME Analysis Card */}
                <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
                  <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
                    {cmePrediction.cme_analysis?.occurring ? (
                      <AlertTriangle className="w-6 h-6 text-red-400" />
                    ) : (
                      <CheckCircle className="w-6 h-6 text-green-400" />
                    )}
                    CME Analysis
                  </h2>

                  {!cmePrediction.success && cmePrediction.error ? (
                    <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-lg p-6 text-center">
                      <XCircle className="w-12 h-12 text-yellow-400 mx-auto mb-4" />
                      <h3 className="text-xl font-semibold text-yellow-300 mb-2">No CME Probability Available</h3>
                      <p className="text-yellow-200 mb-2">{cmePrediction.message || cmePrediction.error}</p>
                      {cmePrediction.cme_analysis?.message && (
                        <p className="text-sm text-yellow-300/80">{cmePrediction.cme_analysis.message}</p>
                      )}
                      {cmePrediction.closest_match_distance && (
                        <p className="text-xs text-gray-400 mt-2">
                          Closest match: {cmePrediction.closest_match_distance.toFixed(2)}° away
                        </p>
                      )}
                    </div>
                  ) : cmePrediction.success && cmePrediction.cme_analysis ? (
                    <div className="space-y-4">
                      {/* Probability and Status */}
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white/5 rounded-lg p-4">
                          <div className="text-sm text-gray-300 mb-1">CME Probability</div>
                          <div className="text-3xl font-bold text-white">
                            {(cmePrediction.cme_analysis.probability * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className={`bg-white/5 rounded-lg p-4 border-2 ${getRiskColor(cmePrediction.cme_analysis.risk_level)}`}>
                          <div className="text-sm text-gray-300 mb-1">Risk Level</div>
                          <div className="text-2xl font-bold">
                            {cmePrediction.cme_analysis.risk_level}
                          </div>
                        </div>
                      </div>

                      {/* Status Badge */}
                      <div className={`p-4 rounded-lg border-2 ${
                        cmePrediction.cme_analysis.occurring
                          ? 'bg-red-500/20 border-red-500/50 text-red-300'
                          : 'bg-green-500/20 border-green-500/50 text-green-300'
                      }`}>
                        <div className="flex items-center gap-2">
                          {cmePrediction.cme_analysis.occurring ? (
                            <XCircle className="w-5 h-5" />
                          ) : (
                            <CheckCircle className="w-5 h-5" />
                          )}
                          <span className="font-semibold">
                            {cmePrediction.cme_analysis.occurring
                              ? 'CME Occurrence Detected'
                              : 'No CME Occurrence Detected'}
                          </span>
                        </div>
                      </div>

                      {/* Score Breakdown */}
                      <div className="bg-white/5 rounded-lg p-4">
                        <h3 className="text-lg font-semibold text-white mb-3">Score Breakdown</h3>
                        <div className="space-y-2">
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-gray-300">Velocity Score</span>
                              <span className="text-white">{(cmePrediction.cme_analysis.scores.velocity_score * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-blue-500 h-2 rounded-full"
                                style={{ width: `${cmePrediction.cme_analysis.scores.velocity_score * 100}%` }}
                              />
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-gray-300">Density Score</span>
                              <span className="text-white">{(cmePrediction.cme_analysis.scores.density_score * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-purple-500 h-2 rounded-full"
                                style={{ width: `${cmePrediction.cme_analysis.scores.density_score * 100}%` }}
                              />
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-gray-300">Temperature Score</span>
                              <span className="text-white">{(cmePrediction.cme_analysis.scores.temperature_score * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-orange-500 h-2 rounded-full"
                                style={{ width: `${cmePrediction.cme_analysis.scores.temperature_score * 100}%` }}
                              />
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-gray-300">Bz Score</span>
                              <span className="text-white">{(cmePrediction.cme_analysis.scores.bz_score * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-red-500 h-2 rounded-full"
                                style={{ width: `${cmePrediction.cme_analysis.scores.bz_score * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-red-300">
                      {cmePrediction.error || 'Failed to analyze CME occurrence'}
                    </div>
                  )}
                </div>

                {/* NOAA Match & Wind Parameters */}
                {cmePrediction.success && cmePrediction.noaa_match && cmePrediction.wind_parameters && (
                  <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
                    <h2 className="text-xl font-semibold text-white mb-4">NOAA Wind Data Match</h2>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-gray-300 uppercase">Match Information</h3>
                        <div className="text-sm">
                          <div className="flex justify-between mb-1">
                            <span className="text-gray-400">Match Distance:</span>
                            <span className="text-white">{cmePrediction.noaa_match.match_distance_degrees.toFixed(4)}°</span>
                          </div>
                          <div className="flex justify-between mb-1">
                            <span className="text-gray-400">NOAA Lat:</span>
                            <span className="text-white">{cmePrediction.noaa_match.latitude.toFixed(4)}°</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">NOAA Lon:</span>
                            <span className="text-white">{cmePrediction.noaa_match.longitude.toFixed(4)}°</span>
                          </div>
                        </div>
                      </div>
                      <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-gray-300 uppercase">Wind Parameters</h3>
                        <div className="text-sm space-y-1">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Speed:</span>
                            <span className="text-white">{cmePrediction.wind_parameters.speed_km_s?.toFixed(2)} km/s</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Density:</span>
                            <span className="text-white">{cmePrediction.wind_parameters.density_particles_cm3?.toFixed(2)} /cm³</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Temperature:</span>
                            <span className="text-white">{cmePrediction.wind_parameters.temperature_k?.toFixed(0)} K</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Bz (GSM):</span>
                            <span className="text-white">{cmePrediction.wind_parameters.bz_gsm_nt?.toFixed(2)} nT</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Bt:</span>
                            <span className="text-white">{cmePrediction.wind_parameters.bt_nt?.toFixed(2)} nT</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            {!loading && !cmePrediction && selectedNoradId && (
              <div className="bg-white/10 backdrop-blur-lg rounded-lg p-12 border border-white/20 text-center text-gray-400">
                Select a satellite to view CME prediction
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

